from typing import List
from abc import ABC, abstractmethod

class LLMProvider(ABC):
    @abstractmethod
    def generate_response(self, prompt: str, max_tokens: int = 500) -> str:
        pass
    
    @abstractmethod
    def generate_embedding(self, text: str) -> List[float]:
        pass

# Groq Implementation
class GroqProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "llama3-8b-8192"):
        import groq
        self.client = groq.Groq(api_key=api_key)
        self.model = model
    
    def generate_response(self, prompt: str, max_tokens: int = 500) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def generate_embedding(self, text: str) -> List[float]:
        # Groq doesn't provide embeddings, so we'll use a simple TF-IDF approach
        return []