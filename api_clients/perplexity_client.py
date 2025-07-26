"""
Perplexity API client
"""
import requests
from .base_client import BaseAPIClient
from config import PERPLEXITY_MODEL, PERPLEXITY_API_ENDPOINT


class PerplexityClient(BaseAPIClient):
    """Client for Perplexity API"""

    def __init__(self, api_key: str, model: str = PERPLEXITY_MODEL):
        super().__init__(api_key)
        self.platform_name = "Perplexity"
        self.model = model

    def call_api(self, prompt: str) -> tuple:
        """Call Perplexity API and return response with sources"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}]
            }
            response = requests.post(
                PERPLEXITY_API_ENDPOINT,
                headers=headers,
                json=data
            )
            if response.status_code == 200:
                response_json = response.json()

                # Extract content
                content = response_json['choices'][0]['message']['content']

                # Extract citations/sources if available
                sources = response_json.get('citations', [])

                # Return both content and sources
                return content, sources
            else:
                return f"Error: {response.status_code} - {response.text}", []
        except Exception as e:
            return self.handle_error(e), []