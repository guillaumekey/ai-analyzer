"""
Gemini API client
"""
import google.generativeai as genai
from .base_client import BaseAPIClient
from config import GEMINI_MODEL


class GeminiClient(BaseAPIClient):
    """Client for Gemini API"""

    def __init__(self, api_key: str, model: str = GEMINI_MODEL):
        super().__init__(api_key)
        self.platform_name = "Gemini"
        self.model = model
        genai.configure(api_key=api_key)

    def call_api(self, prompt: str) -> str:
        """Call Gemini API and return response"""
        try:
            model = genai.GenerativeModel(self.model)
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return self.handle_error(e)