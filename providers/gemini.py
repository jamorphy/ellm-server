from google import genai
from base_provider import BaseProvider
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GeminiProvider(BaseProvider):
    def parse_conversation(self, raw_text: str, system_prompt: str) -> list:
        lines = raw_text.splitlines()
        messages = []
        current_role = None
        current_content = []

        # Skip the first line (model name)
        for line in lines[1:]:
            line = line.strip()
            if line.startswith("<You>:"):
                if current_role and current_content:  # Save previous message
                    messages.append({"role": current_role, "content": " ".join(current_content).strip()})
                current_role = "user"
                current_content = [line[6:].strip()]  # Start new user message
            elif line.startswith("<Assistant>:"):
                if current_role and current_content:  # Save previous message
                    messages.append({"role": current_role, "content": " ".join(current_content).strip()})
                current_role = "assistant"
                current_content = [line[11:].strip()]  # Start new assistant message
            elif line and current_role:  # Continuation of current message
                current_content.append(line.strip())
            elif line:  # Fallback: treat unmarked lines as user input
                if current_role and current_content:
                    messages.append({"role": current_role, "content": " ".join(current_content).strip()})
                current_role = "user"
                current_content = [line]

        # Append the last message if any
        if current_role and current_content:
            messages.append({"role": current_role, "content": " ".join(current_content).strip()})

        return messages

    def generate_stream(self, params: dict, messages: list):
        client = genai.Client(api_key=params["api_key"])

        # Extract history and latest prompt
        history = []
        prompt = None
        for msg in messages:
            if msg["role"] == "user":
                if prompt is not None:  # All prior user messages go to history
                    history.append({"role": "user", "content": prompt})
                prompt = msg["content"]  # Last user message is the prompt
            elif msg["role"] == "assistant":
                history.append({"role": "assistant", "content": msg["content"]})

        if prompt is None or not prompt.strip():  # No user input yet
            prompt = ""

        # Log the history and prompt
        logger.info("Chat history being sent:")
        for msg in history:
            logger.info(f"  Role: {msg['role']}, Content: {msg['content']}")
        logger.info(f"Sending to API - Prompt: '{prompt}'")

        try:
            # Create a new chat session with the full history
            chat = client.chats.create(
                model=params.get("model", "gemini-2.0-flash"),
                history=history
            )
            response = chat.send_message_stream(prompt)
            for chunk in response:
                if chunk.text:
                    yield (chunk.text, None)
        except Exception as e:
            logger.error(f"Gemini API error: {str(e)}")
            yield (f"ERROR: Gemini API failed: {str(e)}", None)
