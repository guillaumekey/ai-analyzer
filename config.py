"""
Configuration file for AI Visibility Audit Tool
"""

# Application settings
APP_NAME = "AI Visibility Audit Tool"
APP_VERSION = "1.0.0"
APP_ICON = "🔍"

# API Models (defaults)
OPENAI_MODEL = "gpt-4o"
PERPLEXITY_MODEL = "sonar"  # Updated to new model system
GEMINI_MODEL = "gemini-1.5-flash"

# Available models for each platform
OPENAI_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-4",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k"
]

# Updated Perplexity models (as of 2025)
PERPLEXITY_MODELS = [
    "sonar",  # Fast, cost-effective search
    "sonar-pro",  # Advanced search with deeper understanding
    "sonar-reasoning",  # Multi-step problem-solving with search
    "sonar-reasoning-pro",  # Enhanced reasoning capabilities
    "sonar-deep-research",  # Exhaustive research and detailed reports
    "sonar-chat"  # Offline chat without search
]

# Updated Gemini models (as of 2025)
GEMINI_MODELS = [
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
    "gemini-1.5-pro",
    "gemini-2.0-flash-exp"
]

# API Endpoints
PERPLEXITY_API_ENDPOINT = "https://api.perplexity.ai/chat/completions"

# Response settings
MAX_TOKENS = 500

# UI Settings
PROMPT_PLACEHOLDER = """What are the best running shoes?
Recommend athletic wear brands
Top sportswear companies"""