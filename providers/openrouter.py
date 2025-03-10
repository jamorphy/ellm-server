import requests
import json
from base_provider import BaseProvider

class OpenRouterProvider(BaseProvider):
    def parse_conversation(self, raw_text: str, system_prompt: str) -> list:
        messages = [{"role": "system", "content": system_prompt}]
        for line in raw_text.splitlines():
            line = line.strip()
            if line.startswith("<You>:"):
                messages.append({"role": "user", "content": line[6:].strip()})
            elif line.startswith("<Assistant>:"):
                messages.append({"role": "assistant", "content": line[11:].strip()})
            elif line:
                messages.append({"role": "user", "content": line})
        return messages

    def generate_stream(self, params: dict, messages: list):
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {params['api_key']}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": params.get("model", "deepseek/deepseek-r1"),
            "messages": messages,
            "stream": True,
            "include_reasoning": params.get("include_reasoning", True),
            "temperature": params.get("temperature", 0.7),
        }
        print(f"Sending payload: {payload}")  # Debug payload
        response = requests.post(url, headers=headers, data=json.dumps(payload), stream=True)
        
        # Handle streaming response manually
        buffer = ""
        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8").strip()
                if line == "data: [DONE]":
                    print("Stream completed")
                    break
                elif line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        if "choices" in data:
                            delta = data["choices"][0]["delta"]
                            content = delta.get("content", "")
                            reasoning = delta.get("reasoning", None)
                            if content or reasoning:
                                yield (content, reasoning)
                        elif "error" in data:
                            yield (f"ERROR: API error: {data['error']}", None)
                    except json.JSONDecodeError as e:
                        print(f"JSON parse error: {e} on line: {line}")
                        yield (f"ERROR: Invalid response chunk: {line}", None)
                    except Exception as e:
                        print(f"Unexpected error: {e} on line: {line}")
                        yield (f"ERROR: Processing error: {str(e)}", None)
                elif line == "data: [DONE]":
                    print("Stream completed")
                    break  # End of stream
