"""
Text analysis utilities
"""
import re
from typing import List, Dict, Any


def count_brand_mentions(text: str, brand_name: str) -> int:
    """Count the number of times the brand is mentioned in the text"""
    if not text or not brand_name:
        return 0
    # Case-insensitive search
    pattern = re.compile(re.escape(brand_name), re.IGNORECASE)
    mentions = pattern.findall(text)
    return len(mentions)


def highlight_brand_mentions(text: str, brand_name: str, color: str = "#FFEB3B") -> str:
    """Highlight brand mentions in text with HTML markup"""
    if not text or not brand_name:
        return text

    # Case-insensitive replacement with highlighting
    pattern = re.compile(re.escape(brand_name), re.IGNORECASE)
    highlighted_text = pattern.sub(
        lambda
            m: f'<span style="background-color: {color}; color: #000; padding: 2px 4px; border-radius: 3px; font-weight: bold;">{m.group()}</span>',
        text
    )
    return highlighted_text


def calculate_visibility_rate(mentions: int, total_prompts: int) -> float:
    """Calculate visibility rate as a percentage"""
    if total_prompts == 0:
        return 0.0
    return (mentions / total_prompts) * 100


def analyze_responses(responses: List[Dict[str, Any]], brand_name: str) -> Dict[str, Any]:
    """Analyze a list of responses for brand mentions"""
    total_mentions = sum(r['mentions'] for r in responses)
    return {
        'total_mentions': total_mentions,
        'visibility_rate': calculate_visibility_rate(total_mentions, len(responses)),
        'responses': responses
    }