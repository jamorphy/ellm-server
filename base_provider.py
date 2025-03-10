from abc import ABC, abstractmethod

class BaseProvider(ABC):
    @abstractmethod
    def parse_conversation(self, raw_text: str, system_prompt: str) -> list:
        pass

    @abstractmethod
    def generate_stream(self, params: dict, messages: list):
        """Yield (content_token, reasoning_token) tuples. Reasoning may be None."""
        pass
