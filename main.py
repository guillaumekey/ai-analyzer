"""
Main Streamlit application for AI Visibility Audit Tool
"""
import streamlit as st
from typing import List, Dict, Any
from config import APP_NAME, APP_ICON, APP_VERSION, PROMPT_PLACEHOLDER
from api_clients import OpenAIClient, PerplexityClient, GeminiClient
from utils import count_brand_mentions, detect_competitors_from_results
from ui import render_sidebar, display_results
from translations import get_text
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title=APP_NAME,
    page_icon=APP_ICON,
    layout="wide"
)

# Initialize session state
if 'api_keys' not in st.session_state:
    st.session_state.api_keys = {
        'openai': '',
        'perplexity': '',
        'gemini': ''
    }

# Initialize session state for results and competitors
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'all_competitors' not in st.session_state:
    st.session_state.all_competitors = []
if 'brand_info' not in st.session_state:
    st.session_state.brand_info = {'name': '', 'url': ''}
if 'prompts_list' not in st.session_state:
    st.session_state.prompts_list = []
if 'language' not in st.session_state:
    st.session_state.language = 'en'
if 'brand_analysis' not in st.session_state:
    st.session_state.brand_analysis = None
if 'vertex_credentials_temp_path' not in st.session_state:
    st.session_state.vertex_credentials_temp_path = None


def test_vertex_in_app(vertex_credentials_path):
    """Test Vertex AI directement dans l'app"""
    st.header("🧪 Test Vertex AI Direct")

    test_text = st.text_area(
        "Test text:",
        value="This is an amazing product! I love it. However, the price is a bit high.",
        height=100
    )

    if st.button("Test Sentiment Analysis"):
        # Test 1: Direct Vertex
        try:
            from utils.vertex_sentiment_analyzer import VertexSentimentAnalyzer

            st.info(f"Loading from: {vertex_credentials_path}")
            st.info(f"File exists: {os.path.exists(vertex_credentials_path)}")

            analyzer = VertexSentimentAnalyzer(vertex_credentials_path)
            st.info(f"Analyzer available: {analyzer.available}")

            if analyzer.available:
                result = analyzer.analyze_sentiment(test_text)
                st.success("✅ Vertex AI Analysis Result:")
                st.json(result)
            else:
                st.error(f"Analyzer not available: {analyzer.error_message}")

        except Exception as e:
            st.error(f"Error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

        # Test 2: Via brand analysis function
        st.subheader("Via analyze_sentiment function")
        try:
            from utils.brand_llm_analysis import analyze_sentiment
            result2 = analyze_sentiment(test_text, analyzer if 'analyzer' in locals() else None)
            st.json(result2)
        except Exception as e:
            st.error(f"Error: {str(e)}")


def validate_inputs(brand_name: str, prompts_text: str, api_keys: Dict[str, str]) -> bool:
    """Validate user inputs"""
    lang = st.session_state.language

    if not brand_name:
        st.error(get_text("error_no_brand", lang))
        return False
    elif not prompts_text:
        st.error(get_text("error_no_prompts", lang))
        return False
    elif not any(api_keys.values()):
        st.error(get_text("error_no_api_key", lang))
        return False
    return True


def process_prompts(prompts: List[str], brand_name: str, api_keys: Dict[str, str], selected_models: Dict[str, str]) -> \
        Dict[str, Any]:
    """Process all prompts across all platforms WITHOUT competitor tracking initially"""
    lang = st.session_state.language

    # Initialize API clients only for platforms with API keys
    clients = {}
    if api_keys.get('openai'):
        clients['chatgpt'] = OpenAIClient(api_keys['openai'], selected_models['openai'])
    if api_keys.get('perplexity'):
        clients['perplexity'] = PerplexityClient(api_keys['perplexity'], selected_models['perplexity'])
    if api_keys.get('gemini'):
        clients['gemini'] = GeminiClient(api_keys['gemini'], selected_models['gemini'])

    # Show which platforms will be tested
    active_platforms = list(clients.keys())
    platform_info = []
    for key, client in clients.items():
        model = selected_models.get(key.replace('chatgpt', 'openai'), 'default')
        platform_info.append(f"{client.platform_name} ({model})")

    st.info(get_text('testing_platforms', lang, count=len(active_platforms), platforms=', '.join(platform_info)))

    # Initialize results WITHOUT competitor tracking
    results = {
        platform: {
            'responses': [],
            'total_mentions': 0,  # Total number of mentions across all prompts
            'unique_mentions': 0  # Number of prompts with at least one mention
        }
        for platform in clients.keys()
    }

    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()

    total_steps = len(prompts) * len(clients)
    current_step = 0

    # Process each prompt
    for i, prompt in enumerate(prompts):
        for platform_key, client in clients.items():
            # Update status
            status_text.text(f"Testing prompt {i + 1}/{len(prompts)} on {client.platform_name}...")

            # Call API
            if platform_key == 'perplexity':
                # Special handling for Perplexity to get sources
                response, sources = client.call_api(prompt)
                mentions = count_brand_mentions(response, brand_name)

                # Store results with sources
                results[platform_key]['responses'].append({
                    'prompt': prompt,
                    'response': response,
                    'mentions': mentions,
                    'sources': sources
                })
            else:
                # Standard handling for other platforms
                response = client.call_api(prompt)
                mentions = count_brand_mentions(response, brand_name)

                # Store results
                results[platform_key]['responses'].append({
                    'prompt': prompt,
                    'response': response,
                    'mentions': mentions
                })

            # Update counts
            results[platform_key]['total_mentions'] += mentions
            if mentions > 0:
                results[platform_key]['unique_mentions'] += 1

            # Update progress
            current_step += 1
            progress_bar.progress(current_step / total_steps)

    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()

    return results


def add_competitor_tracking_to_results(results: Dict[str, Any], competitors: List[str]) -> Dict[str, Any]:
    """Add competitor tracking to existing results without making new API calls"""
    for platform_key, platform_data in results.items():
        # Initialize competitor mentions for the platform
        platform_data['competitor_mentions'] = {comp: {'total': 0, 'unique': 0} for comp in competitors}

        # Count mentions in existing responses
        for response_data in platform_data['responses']:
            response_text = response_data.get('response', '')

            # Add competitor mentions to each response
            response_data['competitor_mentions'] = {}
            for competitor in competitors:
                mentions = count_brand_mentions(response_text, competitor)
                response_data['competitor_mentions'][competitor] = mentions

                # Update platform totals
                platform_data['competitor_mentions'][competitor]['total'] += mentions
                if mentions > 0:
                    platform_data['competitor_mentions'][competitor]['unique'] += 1

    return results


def main():
    """Main application function"""
    lang = st.session_state.language

    # Title and description
    st.title(f"{APP_ICON} {get_text('app_title', lang)}")
    st.markdown(get_text('app_description', lang))

    # Render sidebar and get API keys, models, and vertex credentials path
    api_keys, selected_models, show_individual_results, vertex_credentials_path = render_sidebar()
    st.session_state.api_keys = api_keys

    # Test Vertex AI dans la sidebar
    if st.sidebar.checkbox("🧪 Test Vertex AI Direct", value=False):
        test_vertex_in_app(vertex_credentials_path)
        return  # Stop here to focus on testing

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader(f"📝 {get_text('brand_information', lang)}")
        brand_name = st.text_input(
            get_text('brand_name', lang),
            placeholder=get_text('brand_name_placeholder', lang),
            value=st.session_state.brand_info['name']
        )
        brand_url = st.text_input(
            get_text('brand_url', lang),
            placeholder=get_text('brand_url_placeholder', lang),
            value=st.session_state.brand_info['url']
        )

        # Add competitors field
        competitors_text = st.text_area(
            get_text('competitors', lang),
            height=100,
            placeholder=get_text('competitors_placeholder', lang),
            help=get_text('competitors_help', lang)
        )

    with col2:
        st.subheader(f"🎯 {get_text('test_prompts', lang)}")
        prompts_text = st.text_area(
            get_text('enter_prompts', lang),
            height=150,
            placeholder=PROMPT_PLACEHOLDER
        )

    # Run Analysis Button
    if st.button(f"🚀 {get_text('run_analysis', lang)}", type="primary"):
        if validate_inputs(brand_name, prompts_text, api_keys):
            # Parse prompts
            prompts = [p.strip() for p in prompts_text.split('\n') if p.strip()]

            # Store in session state
            st.session_state.brand_info = {'name': brand_name, 'url': brand_url}
            st.session_state.prompts_list = prompts

            # Parse manual competitors
            manual_competitors = [c.strip() for c in competitors_text.split('\n') if
                                  c.strip()] if competitors_text else []

            # Step 1: Process prompts WITHOUT competitor tracking
            with st.spinner(get_text('analyzing_visibility', lang)):
                results = process_prompts(prompts, brand_name, api_keys, selected_models)

            # Step 2: Auto-detect competitors if OpenAI is configured
            detected_competitors = []
            if api_keys.get('openai'):
                with st.spinner(get_text('detecting_competitors', lang)):
                    ai_client = OpenAIClient(api_keys['openai'], 'gpt-4o')
                    detection_results = detect_competitors_from_results(ai_client, results, brand_name,
                                                                        manual_competitors)
                    detected_competitors = detection_results.get('detected', [])

                    if detected_competitors:
                        st.success(get_text('detected_competitors', lang, competitors=', '.join(detected_competitors)))

            # Step 3: Combine manual and detected competitors
            all_competitors = list(set(manual_competitors + detected_competitors))

            # Step 4: Add competitor tracking to existing results
            if all_competitors:
                with st.spinner(get_text('analyzing_competitors', lang)):
                    results = add_competitor_tracking_to_results(results, all_competitors)
                st.info(get_text('analyzing_count', lang, count=len(all_competitors),
                                 competitors=', '.join(all_competitors)))

            # Store results in session state
            st.session_state.analysis_results = results
            st.session_state.all_competitors = all_competitors

    # Display results if available
    if st.session_state.analysis_results:
        display_results(
            st.session_state.analysis_results,
            st.session_state.brand_info['name'],
            st.session_state.brand_info['url'],
            st.session_state.prompts_list,
            st.session_state.all_competitors,
            show_individual_results
        )

        # Brand LLM Analysis Section
        st.divider()

        # Checkbox for Brand LLM Analysis
        if st.checkbox(
                f"🔍 {get_text('run_brand_analysis', lang)}",
                help=get_text('brand_analysis_help', lang),
                key="enable_brand_analysis"
        ):
            col1, col2, col3 = st.columns([1, 2, 1])

            with col2:
                # Check if Vertex AI credentials exist
                vertex_available = vertex_credentials_path and os.path.exists(vertex_credentials_path)
                if vertex_available:
                    st.info("✅ Google Cloud Natural Language API configured")
                else:
                    st.warning("⚠️ Google Cloud credentials not found. Using alternative sentiment analysis.")

                if st.button(
                        f"▶️ {get_text('run_brand_analysis', lang)}",
                        type="primary",
                        use_container_width=True,
                        key="start_brand_analysis"
                ):
                    # Import the brand analysis module (moved outside try block)
                    try:
                        from utils.brand_llm_analysis import run_brand_llm_analysis, display_brand_llm_analysis
                    except ImportError as e:
                        st.error(f"Error importing brand analysis module: {str(e)}")
                        st.stop()

                    # Create clients for analysis
                    analysis_clients = {}
                    if api_keys.get('openai'):
                        analysis_clients['chatgpt'] = OpenAIClient(api_keys['openai'], selected_models['openai'])
                    if api_keys.get('perplexity'):
                        analysis_clients['perplexity'] = PerplexityClient(api_keys['perplexity'],
                                                                          selected_models['perplexity'])
                    if api_keys.get('gemini'):
                        analysis_clients['gemini'] = GeminiClient(api_keys['gemini'], selected_models['gemini'])

                    if analysis_clients:
                        # Run the brand analysis with Vertex AI credentials if available
                        try:
                            brand_analysis = run_brand_llm_analysis(
                                st.session_state.brand_info['name'],
                                analysis_clients,
                                selected_models,
                                vertex_credentials_path=vertex_credentials_path if vertex_available else None
                            )

                            # Store in session state
                            st.session_state.brand_analysis = brand_analysis
                        except Exception as e:
                            st.error(f"Error during brand analysis: {str(e)}")
                            import traceback
                            st.error(traceback.format_exc())
                    else:
                        st.error(get_text('error_no_api_key', lang))

            # Display brand analysis if available
            if st.session_state.brand_analysis:
                st.divider()
                try:
                    # Ensure the function is imported before using it
                    from utils.brand_llm_analysis import display_brand_llm_analysis
                    display_brand_llm_analysis(
                        st.session_state.brand_analysis,
                        st.session_state.brand_info['name']
                    )
                except Exception as e:
                    st.error(f"Error displaying brand analysis: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())

    # Footer
    st.divider()
    st.markdown(f"{get_text('footer', lang)} | {get_text('app_title', lang)} v{APP_VERSION}")


if __name__ == "__main__":
    main()