"""
Sidebar UI components
"""
import streamlit as st
from typing import Dict, Tuple
import json
import tempfile
import os
from config import (OPENAI_MODELS, PERPLEXITY_MODELS, GEMINI_MODELS,
                    OPENAI_MODEL, PERPLEXITY_MODEL, GEMINI_MODEL)
from translations import get_text


def render_sidebar() -> Tuple[Dict[str, str], Dict[str, str], bool, str]:
    """Render sidebar with API key inputs, model selection and settings"""
    with st.sidebar:
        # Language selector at the top
        col1, col2 = st.columns([1, 1])
        with col1:
            lang = st.selectbox(
                "🌐",
                options=["en", "fr"],
                format_func=lambda x: {"en": "🇬🇧 English", "fr": "🇫🇷 Français"}[x],
                key="language",
                label_visibility="collapsed"
            )

        st.header(f"⚙️ {get_text('configuration', lang)}")

        # API Keys Section
        st.subheader(f"🔑 {get_text('api_keys', lang)}")

        api_keys = {}
        selected_models = {}

        # OpenAI
        with st.expander("OpenAI / ChatGPT", expanded=True):
            api_keys['openai'] = st.text_input(
                get_text('api_key', lang),
                value=st.session_state.get('api_keys', {}).get('openai', ''),
                type="password",
                key="openai_key",
                help=get_text('enter_api_key', lang, platform='OpenAI')
            )
            if api_keys['openai']:
                selected_models['openai'] = st.selectbox(
                    get_text('model', lang),
                    options=OPENAI_MODELS,
                    index=OPENAI_MODELS.index(OPENAI_MODEL),
                    key="openai_model",
                    help=get_text('select_model', lang, platform='OpenAI')
                )
            else:
                selected_models['openai'] = OPENAI_MODEL

        # Perplexity
        with st.expander("Perplexity", expanded=True):
            api_keys['perplexity'] = st.text_input(
                get_text('api_key', lang),
                value=st.session_state.get('api_keys', {}).get('perplexity', ''),
                type="password",
                key="perplexity_key",
                help=get_text('enter_api_key', lang, platform='Perplexity')
            )
            if api_keys['perplexity']:
                # Model descriptions in both languages
                model_help = {
                    "en": """
                    • sonar: Fast, cost-effective search (recommended)
                    • sonar-pro: Advanced search with deeper understanding
                    • sonar-reasoning: Multi-step problem-solving
                    • sonar-deep-research: Exhaustive research & reports
                    • sonar-chat: Offline chat without search
                    """,
                    "fr": """
                    • sonar: Recherche rapide et économique (recommandé)
                    • sonar-pro: Recherche avancée avec compréhension approfondie
                    • sonar-reasoning: Résolution de problèmes multi-étapes
                    • sonar-deep-research: Recherche exhaustive et rapports
                    • sonar-chat: Chat hors ligne sans recherche
                    """
                }
                selected_models['perplexity'] = st.selectbox(
                    get_text('model', lang),
                    options=PERPLEXITY_MODELS,
                    index=PERPLEXITY_MODELS.index(PERPLEXITY_MODEL),
                    key="perplexity_model",
                    help=model_help[lang]
                )
            else:
                selected_models['perplexity'] = PERPLEXITY_MODEL

        # Gemini
        with st.expander("Google Gemini", expanded=True):
            api_keys['gemini'] = st.text_input(
                get_text('api_key', lang),
                value=st.session_state.get('api_keys', {}).get('gemini', ''),
                type="password",
                key="gemini_key",
                help=get_text('enter_api_key', lang, platform='Google Gemini')
            )
            if api_keys['gemini']:
                selected_models['gemini'] = st.selectbox(
                    get_text('model', lang),
                    options=GEMINI_MODELS,
                    index=GEMINI_MODELS.index(GEMINI_MODEL),
                    key="gemini_model",
                    help=get_text('select_model', lang, platform='Gemini')
                )
            else:
                selected_models['gemini'] = GEMINI_MODEL

        # Google Cloud / Vertex AI Section
        st.divider()
        st.subheader("☁️ Google Cloud (Sentiment Analysis)")

        vertex_credentials_path = None

        # Check if we have stored credentials in session state
        if 'vertex_credentials_temp_path' in st.session_state and st.session_state.vertex_credentials_temp_path:
            if os.path.exists(st.session_state.vertex_credentials_temp_path):
                vertex_credentials_path = st.session_state.vertex_credentials_temp_path
                st.success("✅ Google Cloud credentials loaded")

                # Option to remove credentials
                if st.button("🗑️ Remove credentials", key="remove_vertex_creds"):
                    try:
                        os.remove(st.session_state.vertex_credentials_temp_path)
                        st.session_state.vertex_credentials_temp_path = None
                        st.rerun()
                    except:
                        pass

        # Upload new credentials
        uploaded_file = st.file_uploader(
            "Upload Google Cloud credentials JSON",
            type=['json'],
            key="vertex_upload",
            help="Upload your Google Cloud service account JSON file for sentiment analysis"
        )

        if uploaded_file is not None:
            try:
                # Validate JSON structure
                json_content = json.loads(uploaded_file.read())

                # Check for required fields
                required_fields = ['type', 'project_id', 'private_key', 'client_email']
                if all(field in json_content for field in required_fields):
                    # Create a temporary file
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
                        json.dump(json_content, tmp_file)
                        vertex_credentials_path = tmp_file.name
                        # Store in session state
                        st.session_state.vertex_credentials_temp_path = vertex_credentials_path

                    st.success("✅ Valid Google Cloud credentials uploaded!")

                    # Show project info
                    st.info(f"Project ID: {json_content.get('project_id')}")
                else:
                    st.error("❌ Invalid credentials file. Missing required fields.")

            except json.JSONDecodeError:
                st.error("❌ Invalid JSON file")
            except Exception as e:
                st.error(f"❌ Error reading credentials: {str(e)}")

        # Alternative: Use environment variable
        if not vertex_credentials_path:
            env_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
            if env_path and os.path.exists(env_path):
                vertex_credentials_path = env_path
                st.info("ℹ️ Using credentials from environment variable")

        # Show configured platforms
        st.divider()
        st.subheader(f"🟢 {get_text('active_platforms', lang)}")

        platforms_status = {
            'ChatGPT': bool(api_keys['openai']),
            'Perplexity': bool(api_keys['perplexity']),
            'Gemini': bool(api_keys['gemini'])
        }

        for platform, is_active in platforms_status.items():
            if is_active:
                model_key = platform.lower().replace('chatgpt', 'openai')
                st.success(f"✅ {platform} ({selected_models.get(model_key, 'default')})")
            else:
                st.info(f"⚪ {platform} ({get_text('no_api_key', lang)})")

        # Show Vertex AI status
        if vertex_credentials_path:
            st.success("✅ Google Cloud NL API (Sentiment Analysis)")
        else:
            st.info("⚪ Google Cloud NL API (Optional)")

        active_count = sum(platforms_status.values())
        if active_count == 0:
            st.warning(f"⚠️ {get_text('no_platforms', lang)}")
        else:
            st.info(f"📊 {get_text('platforms_ready', lang, count=active_count)}")

        st.divider()

        # Display options
        st.subheader(get_text('display_options', lang))
        show_individual_results = st.checkbox(
            get_text('show_individual', lang),
            value=True,
            help=get_text('display_detailed_help', lang)
        )

        return api_keys, selected_models, show_individual_results, vertex_credentials_path