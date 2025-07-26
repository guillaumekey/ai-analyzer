"""
Export utilities - Enhanced version with complete data export
"""
import json
from datetime import datetime
from typing import Dict, Any, List, Optional


def prepare_export_data(
        brand_name: str,
        brand_url: str,
        prompts: List[str],
        results: Dict[str, Any],
        unique_mentions: int,
        total_mentions: int,
        total_queries: int,
        competitors: Optional[List[str]] = None,
        brand_analysis: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Prepare complete data for export including all analysis results"""
    from .text_analysis import calculate_visibility_rate

    # Basic export structure
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

    # Add Perplexity sources if available
    if 'perplexity' in results:
        all_sources = []
        for response in results['perplexity']['responses']:
            if 'sources' in response and response['sources']:
                all_sources.extend(response['sources'])

        if all_sources:
            # Create source statistics
            from collections import Counter
            from urllib.parse import urlparse

            domain_counts = Counter()
            unique_urls = set(all_sources)

            for url in all_sources:
                try:
                    parsed = urlparse(url)
                    domain = parsed.netloc or url
                    domain_counts[domain] += 1
                except:
                    domain_counts[url] += 1

            export_data['perplexity_sources'] = {
                'total_sources': len(all_sources),
                'unique_urls': len(unique_urls),
                'unique_domains': len(domain_counts),
                'domain_frequency': dict(domain_counts.most_common()),
                'all_sources': list(unique_urls)
            }

    # Add competitor analysis if available
    if competitors and any('competitor_mentions' in results.get(p, {}) for p in results):
        competitor_summary = {}

        # Calculate totals for each competitor across all platforms
        for competitor in competitors:
            total_unique = 0
            total_mentions = 0
            platform_details = {}

            for platform in results:
                if 'competitor_mentions' in results[platform]:
                    comp_data = results[platform]['competitor_mentions'].get(competitor, {})
                    platform_unique = comp_data.get('unique', 0)
                    platform_total = comp_data.get('total', 0)

                    total_unique += platform_unique
                    total_mentions += platform_total

                    platform_details[platform] = {
                        'unique': platform_unique,
                        'total': platform_total
                    }

            competitor_summary[competitor] = {
                'total_unique_mentions': total_unique,
                'total_all_mentions': total_mentions,
                'visibility_rate': calculate_visibility_rate(total_unique, total_queries),
                'platform_breakdown': platform_details
            }

        # Add brand comparison
        export_data['competitor_analysis'] = {
            'competitors_analyzed': competitors,
            'competitor_metrics': competitor_summary,
            'brand_vs_competitors': {
                'brand_total_unique': unique_mentions,
                'brand_total_mentions': total_mentions,
                'brand_visibility_rate': calculate_visibility_rate(unique_mentions, total_queries),
                'ranking': _calculate_ranking(brand_name, unique_mentions, competitor_summary)
            }
        }

    # Add brand LLM analysis if available
    if brand_analysis:
        llm_analysis_data = {
            'analysis_performed': True,
            'overall_sentiment': brand_analysis.get('overall_sentiment', {}),
            'sentiment_by_platform': {},
            'key_themes': brand_analysis.get('key_themes', []),
            'brand_aspects': brand_analysis.get('brand_aspects', {}),
            'entities_detected': brand_analysis.get('entities', [])
        }

        # Add platform-specific sentiment data
        for platform, sentiment_data in brand_analysis.get('sentiment_analysis', {}).items():
            llm_analysis_data['sentiment_by_platform'][platform] = {
                'polarity': sentiment_data.get('polarity', 0),
                'magnitude': sentiment_data.get('magnitude', 0),
                'sentiment': sentiment_data.get('sentiment', 'neutral'),
                'positive_aspects_count': sentiment_data.get('positive_count',
                                                             len(sentiment_data.get('positive_aspects', []))),
                'negative_aspects_count': sentiment_data.get('negative_count',
                                                             len(sentiment_data.get('negative_aspects', []))),
                'positive_aspects': sentiment_data.get('positive_aspects', [])[:5],  # Top 5
                'negative_aspects': sentiment_data.get('negative_aspects', [])[:5],  # Top 5
                'api_used': sentiment_data.get('api_used', 'unknown')
            }

        # Add global sentiment metrics
        total_positive = sum(data.get('positive_aspects_count', 0)
                             for data in llm_analysis_data['sentiment_by_platform'].values())
        total_negative = sum(data.get('negative_aspects_count', 0)
                             for data in llm_analysis_data['sentiment_by_platform'].values())

        llm_analysis_data['global_sentiment_metrics'] = {
            'total_positive_aspects': total_positive,
            'total_negative_aspects': total_negative,
            'positive_negative_ratio': total_positive / total_negative if total_negative > 0 else total_positive,
            'overall_polarity': brand_analysis.get('overall_sentiment', {}).get('polarity', 0),
            'overall_sentiment_label': brand_analysis.get('overall_sentiment', {}).get('sentiment', 'neutral')
        }

        # Add raw responses if needed for reference
        llm_analysis_data['analysis_prompts'] = {
            'knowledge_responses': _summarize_responses(brand_analysis.get('knowledge_responses', {})),
            'reputation_responses': _summarize_responses(brand_analysis.get('reputation_responses', {}))
        }

        export_data['brand_llm_analysis'] = llm_analysis_data
    else:
        export_data['brand_llm_analysis'] = {'analysis_performed': False}

    # Add metadata
    export_data['metadata'] = {
        'export_version': '2.0',
        'export_timestamp': datetime.now().isoformat(),
        'analysis_configuration': {
            'platforms_used': list(results.keys()),
            'prompts_tested': prompts,
            'competitors_tracked': competitors or [],
            'llm_analysis_included': brand_analysis is not None
        }
    }

    return export_data


def _calculate_ranking(brand_name: str, brand_unique: int, competitor_summary: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate brand ranking compared to competitors"""
    # Create ranking list
    ranking_list = [(brand_name, brand_unique)]

    for comp_name, comp_data in competitor_summary.items():
        ranking_list.append((comp_name, comp_data['total_unique_mentions']))

    # Sort by unique mentions (descending)
    ranking_list.sort(key=lambda x: x[1], reverse=True)

    # Find brand position
    brand_position = next(i for i, (name, _) in enumerate(ranking_list, 1) if name == brand_name)

    return {
        'position': brand_position,
        'total_brands': len(ranking_list),
        'top_3': [{'name': name, 'mentions': mentions} for name, mentions in ranking_list[:3]],
        'gap_to_leader': ranking_list[0][1] - brand_unique if brand_position > 1 else 0
    }


def _summarize_responses(responses_dict: Dict[str, List[str]]) -> Dict[str, Dict[str, Any]]:
    """Summarize LLM responses for export"""
    summary = {}

    for platform, responses in responses_dict.items():
        if responses:
            # Join all responses and get word count
            full_text = ' '.join(responses)
            word_count = len(full_text.split())

            summary[platform] = {
                'response_count': len(responses),
                'total_word_count': word_count,
                'average_response_length': word_count // len(responses) if responses else 0,
                # Include first 200 chars of each response as preview
                'response_previews': [resp[:200] + '...' if len(resp) > 200 else resp for resp in responses]
            }

    return summary


def export_to_json(data: Dict[str, Any], brand_name: str) -> tuple:
    """Export data to JSON string with proper formatting"""
    filename = f"{brand_name}_ai_visibility_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # Pretty print JSON with proper indentation
    json_str = json.dumps(data, indent=2, ensure_ascii=False)

    return json_str, filename


def export_brand_analysis_separately(brand_analysis: Dict[str, Any], brand_name: str) -> tuple:
    """Export brand analysis data separately if needed (legacy support)"""
    export_data = {
        'brand_name': brand_name,
        'analysis_date': datetime.now().isoformat(),
        'overall_sentiment': brand_analysis.get('overall_sentiment', {}),
        'platform_sentiments': brand_analysis.get('sentiment_analysis', {}),
        'key_themes': brand_analysis.get('key_themes', []),
        'brand_aspects': brand_analysis.get('brand_aspects', {}),
        'entities': brand_analysis.get('entities', [])
    }

    filename = f"{brand_name}_brand_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    json_str = json.dumps(export_data, indent=2, ensure_ascii=False)

    return json_str, filename