from anthropic import Anthropic
from base_provider import BaseProvider

class AnthropicProvider(BaseProvider):
    def parse_conversation(self, raw_text: str, system_prompt: str) -> list:
        messages = []
        for line in raw_text.splitlines():
            line = line.strip()
            if line.startswith("<You>:"):
                messages.append({"role": "user", "content": line[6:].strip()})
            elif line.startswith("<Assistant>:"):
                messages.append({"role": "assistant", "content": line[11:].strip()})
            elif line:  # Fallback: treat unmarked lines as user input
                messages.append({"role": "user", "content": line})
        return messages

    def generate_stream(self, params: dict, messages: list):
        client = Anthropic(api_key=params["api_key"])
        stream = client.messages.create(
            model=params.get("model", "claude-3-5-sonnet-20241022"),
            system=params.get("system_prompt", "You are a helpful assistant."),
            messages=messages,
            stream=True,
            temperature=params.get("temperature", 0.7),
            max_tokens=params.get("max_tokens", 1024),
        )
        for event in stream:
            if event.type == "content_block_delta" and hasattr(event.delta, "text"):
                yield (event.delta.text, None)
