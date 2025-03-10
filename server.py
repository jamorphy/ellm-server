import socketserver
from config import load_config
from providers.openai import OpenAIProvider
from providers.openrouter import OpenRouterProvider
from providers.anthropic import AnthropicProvider  # Assuming this exists
import traceback
import os

from dotenv import load_dotenv
load_dotenv()

provider_classes = {
    "openai": OpenAIProvider,
    "openrouter": OpenRouterProvider,
    "anthropic": AnthropicProvider,
}

class ChatRequestHandler(socketserver.StreamRequestHandler):
    def handle(self):
        try:
            model_name = self.rfile.readline().decode("utf-8").strip()
            if not model_name:
                self.wfile.write(b"ERROR: No model specified\n")
                return

            config = load_config()
            provider_name = None
            provider_config = None

            # Search for the model in nested 'models' structure
            for p_name, p_config in config["providers"].items():
                if "models" in p_config:
                    for m_name, m_config in p_config["models"].items():
                        if model_name == m_name:
                            provider_name = p_name
                            provider_config = m_config.copy()
                            provider_config["model"] = model_name  # Ensure model is set
                            break
                if provider_config:
                    break

            if not provider_name or not provider_config:
                self.wfile.write(f"ERROR: Model '{model_name}' not found in config\n".encode())
                return

            provider_class = provider_classes.get(provider_name)
            if not provider_class:
                self.wfile.write(f"ERROR: Unknown provider '{provider_name}' for model '{model_name}'\n".encode())
                return

            raw_text = self.rfile.read().decode("utf-8").strip()
            if not raw_text:
                self.wfile.write(b"ERROR: No message provided\n")
                return

            provider = provider_class()
            messages = provider.parse_conversation(raw_text, provider_config["system_prompt"])

            try:
                for content, reasoning in provider.generate_stream(provider_config, messages):
                    if content:
                        self.wfile.write(content.encode())
                        self.wfile.flush()
                    if reasoning:
                        self.wfile.write(reasoning.encode())
                        self.wfile.flush()
            except Exception as stream_error:
                error_msg = f"ERROR: Streaming failed for model '{model_name}': {str(stream_error)}\n"
                print(error_msg)
                print(traceback.format_exc())
                self.wfile.write(error_msg.encode())

        except Exception as e:
            error_msg = f"ERROR: Server error for model '{model_name}': {str(e)}\n"
            print(error_msg)
            print(traceback.format_exc())
            self.wfile.write(error_msg.encode())

    def finish(self):
        self.wfile.flush()
        super().finish()

if __name__ == "__main__":
    HOST, PORT = "localhost", 9999
    server = socketserver.ThreadingTCPServer((HOST, PORT), ChatRequestHandler)
    print(f"Server running on {HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped")
        server.server_close()
