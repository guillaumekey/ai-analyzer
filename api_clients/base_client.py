"""
Base API client class
"""
from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseAPIClient(ABC):
    """Abstract base class for API clients"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.platform_name = ""

    @abstractmethod
    def call_api(self, prompt: str) -> str:
        """Make API call and return response"""
        pass

    def handle_error(self, error: Exception) -> str:
        """Standard error handling"""
        return f"Error on {self.platform_name}: {str(error)}"