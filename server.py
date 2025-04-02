import socketserver
from config import load_config
from providers.openai import OpenAIProvider
from providers.openrouter import OpenRouterProvider
from providers.anthropic import AnthropicProvider
from providers.gemini import GeminiProvider
import traceback
import os
from dotenv import load_dotenv

load_dotenv()

provider_classes = {
    "openai": OpenAIProvider,
    "openrouter": OpenRouterProvider,
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
}

class ChatRequestHandler(socketserver.StreamRequestHandler):
    def handle(self):
        try:
            first_line = self.rfile.readline().decode("utf-8").strip()
            if first_line == "list-models":
                config = load_config()
                models = []
                for provider_name, provider_config in config["providers"].items():
                    if "models" in provider_config:
                        models.extend(provider_config["models"].keys())
                response = "\n".join(models) + "\n"
                self.wfile.write(response.encode())
                return

            # Normal chat handling
            if not first_line:
                self.wfile.write(b"ERROR: No model specified\n")
                return

            config = load_config()
            provider_name = None
            provider_config = None

            for p_name, p_config in config["providers"].items():
                if "models" in p_config:
                    for m_name, m_config in p_config["models"].items():
                        if first_line == m_name:
                            provider_name = p_name
                            provider_config = m_config.copy()
                            provider_config["model"] = first_line
                            break
                if provider_config:
                    break

            if not provider_name or not provider_config:
                self.wfile.write(f"ERROR: Model '{first_line}' not found in config\n".encode())
                return

            provider_class = provider_classes.get(provider_name)
            if not provider_class:
                self.wfile.write(f"ERROR: Unknown provider '{provider_name}' for model '{first_line}'\n".encode())
                return

            raw_text = self.rfile.read().decode("utf-8").strip()
            if not raw_text:
                self.wfile.write(b"ERROR: No message provided\n")
                return

            provider = provider_class()
            messages = provider.parse_conversation(raw_text, provider_config["system_prompt"])

            for content, reasoning in provider.generate_stream(provider_config, messages):
                if content:
                    self.wfile.write(content.encode())
                    self.wfile.flush()
                if reasoning:
                    self.wfile.write(reasoning.encode())
                    self.wfile.flush()

        except Exception as e:
            error_msg = f"ERROR: Server error: {str(e)}\n"
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
