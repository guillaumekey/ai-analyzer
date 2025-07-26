"""
Utils package initialization
"""
from .text_analysis import (
    count_brand_mentions,
    highlight_brand_mentions,
    calculate_visibility_rate,
    analyze_responses
)

from .export_utils import prepare_export_data, export_to_json
from .competitor_detection import detect_competitors_from_results

# Import from ui folder instead of utils
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ui.results_display import (
    display_summary_metrics,
    display_platform_breakdown,
    display_sources_analysis,
    display_detailed_results,
    display_results
)
from ui.sidebar import render_sidebar
from ui.competitor_visualizations import display_competitor_analysis_section

# PDF export (optional)
try:
    from .pdf_export import generate_pdf_report
except ImportError:
    pass

__all__ = [
    'count_brand_mentions',
    'highlight_brand_mentions',
    'calculate_visibility_rate',
    'analyze_responses',
    'prepare_export_data',
    'export_to_json',
    'detect_competitors_from_results',
    'display_summary_metrics',
    'display_platform_breakdown',
    'display_sources_analysis',
    'display_detailed_results',
    'display_results',
    'render_sidebar',
    'display_competitor_analysis_section',
    'generate_pdf_report'
]