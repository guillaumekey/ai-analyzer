"""
Results display UI components
"""
import streamlit as st
from typing import Dict, Any, List
import pandas as pd
from collections import Counter
from urllib.parse import urlparse
from datetime import datetime
from .competitor_visualizations import display_competitor_analysis_section
from translations import get_text


def display_summary_metrics(unique_mentions: int, total_mentions: int, total_queries: int, num_prompts: int,
                            num_platforms: int):
    """Display summary metrics with unique and total mentions"""
    from utils import calculate_visibility_rate
    lang = st.session_state.get('language', 'en')

    st.subheader(f"📊 {get_text('visibility_summary', lang)}")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            get_text('unique_mentions', lang),
            unique_mentions,
            help=get_text('unique_mentions', lang)
        )

    with col2:
        st.metric(
            get_text('total_mentions', lang),
            total_mentions,
            help=get_text('total_mentions', lang)
        )

    with col3:
        st.metric(
            get_text('visibility_rate', lang),
            f"{calculate_visibility_rate(unique_mentions, total_queries):.1f}%",
            help=get_text('visibility_rate', lang)
        )

    with col4:
        st.metric(get_text('prompts_tested', lang), num_prompts)

    with col5:
        st.metric(get_text('platforms', lang), num_platforms)


def display_platform_breakdown(results: Dict[str, Any], num_prompts: int):
    """Display platform-specific breakdown with unique and total mentions"""
    from utils import calculate_visibility_rate
    lang = st.session_state.get('language', 'en')

    st.subheader(f"🎯 {get_text('platform_breakdown', lang)}")

    # Get active platforms from results
    active_platforms = list(results.keys())
    num_platforms = len(active_platforms)

    # Create columns based on number of active platforms
    if num_platforms > 0:
        platform_cols = st.columns(num_platforms)

        platform_names = {
            'chatgpt': 'ChatGPT',
            'perplexity': 'Perplexity',
            'gemini': 'Gemini'
        }

        for idx, platform_key in enumerate(active_platforms):
            with platform_cols[idx]:
                unique_mentions = results[platform_key]['unique_mentions']
                total_mentions = results[platform_key]['total_mentions']
                visibility_rate = calculate_visibility_rate(unique_mentions, num_prompts)

                st.markdown(f"### {platform_names.get(platform_key, platform_key)}")
                st.metric(get_text('unique_mentions', lang), unique_mentions)
                st.metric(get_text('total_mentions', lang), total_mentions)
                st.metric(get_text('visibility_rate', lang), f"{visibility_rate:.1f}%")

                # Add mention density
                if unique_mentions > 0:
                    density = total_mentions / unique_mentions
                    st.metric(get_text('avg_per_prompt', lang), f"{density:.1f}")


def display_sources_analysis(results: Dict[str, Any]):
    """Display sources analysis for Perplexity with clickable links"""
    lang = st.session_state.get('language', 'en')

    if 'perplexity' in results:
        st.subheader(f"🔗 {get_text('sources_analysis', lang)}")

        # Collect all sources with their full URLs
        all_sources = []
        source_to_url = {}  # Map domain to full URLs

        for response in results['perplexity']['responses']:
            if 'sources' in response and response['sources']:
                for source_url in response['sources']:
                    all_sources.append(source_url)

                    # Parse domain from URL
                    try:
                        parsed = urlparse(source_url)
                        domain = parsed.netloc or source_url

                        # Store the full URL for each domain
                        if domain not in source_to_url:
                            source_to_url[domain] = []
                        source_to_url[domain].append(source_url)
                    except:
                        domain = source_url
                        if domain not in source_to_url:
                            source_to_url[domain] = []
                        source_to_url[domain].append(source_url)

        if all_sources:
            # Count domain occurrences
            domain_counts = Counter()
            for source_url in all_sources:
                try:
                    parsed = urlparse(source_url)
                    domain = parsed.netloc or source_url
                    domain_counts[domain] += 1
                except:
                    domain_counts[source_url] += 1

            # Create DataFrame for display
            data_for_df = []
            for domain, count in domain_counts.most_common():
                # Get unique URLs for this domain
                unique_urls = list(set(source_to_url.get(domain, [])))

                # Create entry for each unique URL
                if len(unique_urls) == 1:
                    # Single URL for this domain
                    data_for_df.append({
                        get_text('domain', lang): domain,
                        get_text('count', lang): count,
                        get_text('source_url', lang): unique_urls[0]
                    })
                else:
                    # Multiple URLs for this domain - group them
                    data_for_df.append({
                        get_text('domain', lang): domain,
                        get_text('count', lang): count,
                        get_text('source_url', lang): get_text('different_pages', lang, count=len(unique_urls))
                    })

            df_sources = pd.DataFrame(data_for_df)

            col1, col2 = st.columns([3, 1])

            with col1:
                # Display the main table
                st.dataframe(df_sources, use_container_width=True)

                # Expandable section for domains with multiple URLs
                domains_with_multiple = [d for d, urls in source_to_url.items() if len(set(urls)) > 1]

                if domains_with_multiple:
                    with st.expander(f"📋 {get_text('view_all_urls', lang)}"):
                        for domain in domains_with_multiple:
                            if domain in source_to_url:
                                st.markdown(f"**{domain}:**")
                                unique_urls = list(set(source_to_url[domain]))
                                for idx, url in enumerate(unique_urls, 1):
                                    st.markdown(f"{idx}. [{url}]({url})")
                                st.markdown("---")

            with col2:
                st.metric(get_text('total_sources', lang), len(all_sources))
                st.metric(get_text('unique_domains', lang), len(domain_counts))
                st.metric(get_text('unique_urls', lang), len(set(all_sources)))

            # Add download button for full source list
            if st.button(f"📥 {get_text('export_sources', lang)}"):
                # Create a detailed export
                export_data = []
                for domain, urls in source_to_url.items():
                    for url in set(urls):
                        export_data.append({
                            get_text('domain', lang): domain,
                            'Full URL': url,
                            get_text('count', lang): urls.count(url)
                        })

                export_df = pd.DataFrame(export_data)
                csv = export_df.to_csv(index=False)

                st.download_button(
                    label=get_text('download_csv', lang),
                    data=csv,
                    file_name="perplexity_sources.csv",
                    mime="text/csv"
                )

        else:
            st.info(get_text('no_sources', lang))


def display_detailed_results(results: Dict[str, Any], brand_name: str):
    """Display detailed results for each platform with brand highlighting"""
    from utils.text_analysis import highlight_brand_mentions
    lang = st.session_state.get('language', 'en')

    st.subheader(f"📝 {get_text('detailed_results', lang)}")

    # Create tabs only for active platforms
    active_platforms = list(results.keys())
    platform_names = {
        'chatgpt': 'ChatGPT',
        'perplexity': 'Perplexity',
        'gemini': 'Gemini'
    }

    tab_names = [platform_names.get(p, p) for p in active_platforms]
    tabs = st.tabs(tab_names)

    for idx, (tab, platform_key) in enumerate(zip(tabs, active_platforms)):
        with tab:
            for i, result in enumerate(results[platform_key]['responses']):
                with st.expander(f"{get_text('prompt', lang)} {i + 1}: {result['prompt'][:50]}..."):
                    st.write(f"**{get_text('mentions_found', lang)}** {result['mentions']}")

                    # Highlight brand mentions in response
                    highlighted_response = highlight_brand_mentions(result['response'], brand_name)
                    st.write(f"**{get_text('response', lang)}**")
                    st.markdown(highlighted_response, unsafe_allow_html=True)

                    # Show sources if available (Perplexity)
                    if 'sources' in result and result['sources']:
                        st.write(f"**{get_text('sources', lang)}**")
                        for j, source in enumerate(result['sources'], 1):
                            st.write(f"{j}. {source}")


def display_results(results: Dict[str, Any], brand_name: str, brand_url: str,
                    prompts: List[str], competitors: List[str], show_individual: bool):
    """Display all results with unique and total mentions"""
    from utils import prepare_export_data
    from utils.export_utils import export_to_json

    lang = st.session_state.get('language', 'en')

    st.success(f"✅ {get_text('analysis_complete', lang)}")

    # Calculate totals based on active platforms
    active_platforms = list(results.keys())

    # Calculate unique and total mentions
    total_unique_mentions = sum(results[p]['unique_mentions'] for p in active_platforms)
    total_all_mentions = sum(results[p]['total_mentions'] for p in active_platforms)
    total_queries = len(prompts) * len(active_platforms)

    # Display metrics
    display_summary_metrics(total_unique_mentions, total_all_mentions, total_queries, len(prompts),
                            len(active_platforms))

    # Platform breakdown
    display_platform_breakdown(results, len(prompts))

    # Competitive Analysis Section
    if competitors:
        display_competitor_analysis_section(results, brand_name, competitors)

    # Detailed results
    if show_individual and active_platforms:
        display_detailed_results(results, brand_name)

    # Export section - JSON only
    st.subheader(f"💾 {get_text('export_results', lang)}")

    # Get brand analysis from session state if available
    brand_analysis = st.session_state.get('brand_analysis', None)

    export_data = prepare_export_data(
        brand_name=brand_name,
        brand_url=brand_url,
        prompts=prompts,
        results=results,
        unique_mentions=total_unique_mentions,
        total_mentions=total_all_mentions,
        total_queries=total_queries,
        competitors=competitors,
        brand_analysis=brand_analysis
    )

    json_str, filename = export_to_json(export_data, brand_name)

    st.download_button(
        label=f"📥 {get_text('download_report', lang)}",
        data=json_str,
        file_name=filename,
        mime="application/json"
    )

    # Sources analysis AFTER export (only if Perplexity was used)
    if 'perplexity' in results:
        display_sources_analysis(results)