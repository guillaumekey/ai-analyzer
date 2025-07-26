"""
Export utilities
"""
import json
from datetime import datetime
from typing import Dict, Any, List


def prepare_export_data(
        brand_name: str,
        brand_url: str,
        prompts: List[str],
        results: Dict[str, Any],
        unique_mentions: int,
        total_mentions: int,
        total_queries: int
) -> Dict[str, Any]:
    """Prepare data for export with unique and total mentions"""
    from .text_analysis import calculate_visibility_rate

    export_data = {
        'summary': {
            'brand_name': brand_name,
            'brand_url': brand_url,
            'analysis_date': datetime.now().isoformat(),
            'total_prompts': len(prompts),
            'unique_mentions': unique_mentions,
            'total_mentions': total_mentions,
            'overall_visibility_rate': calculate_visibility_rate(unique_mentions, total_queries),
            'mention_density': total_mentions / unique_mentions if unique_mentions > 0 else 0
        },
        'platform_summary': {},
        'detailed_results': results
    }

    # Add platform summaries with unique and total mentions
    for platform in ['chatgpt', 'perplexity', 'gemini']:
        if platform in results:
            unique = results[platform]['unique_mentions']
            total = results[platform]['total_mentions']
            export_data['platform_summary'][platform] = {
                'unique_mentions': unique,
                'total_mentions': total,
                'visibility_rate': calculate_visibility_rate(unique, len(prompts)),
                'mention_density': total / unique if unique > 0 else 0
            }

    return export_data


def export_to_json(data: Dict[str, Any], brand_name: str) -> tuple:
    """Export data to JSON string"""
    filename = f"{brand_name}_ai_visibility_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    return json.dumps(data, indent=2), filename