"""
OpenAI API client
"""
from openai import OpenAI
from .base_client import BaseAPIClient
from config import OPENAI_MODEL, MAX_TOKENS


class OpenAIClient(BaseAPIClient):
    """Client for OpenAI API"""

    def __init__(self, api_key: str, model: str = OPENAI_MODEL):
        super().__init__(api_key)
        self.platform_name = "ChatGPT"
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def call_api(self, prompt: str) -> str:
        """Call OpenAI API and return response"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=MAX_TOKENS
            )
            return response.choices[0].message.content
        except Exception as e:
            return self.handle_error(e)