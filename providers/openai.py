import openai
from base_provider import BaseProvider

class OpenAIProvider(BaseProvider):
    def parse_conversation(self, raw_text: str, system_prompt: str) -> list:
        messages = [{"role": "system", "content": system_prompt}]
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
        client = openai.OpenAI(api_key=params["api_key"])
        stream = client.chat.completions.create(
            model=params.get("model", "gpt-4o"),
            messages=messages,
            stream=True,
            temperature=params.get("temperature", 0.7),
        )
        for chunk in stream:
            if content := chunk.choices[0].delta.content:
                if content:
                    yield (content, None)
