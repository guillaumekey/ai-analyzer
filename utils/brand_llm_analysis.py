"""
Brand LLM Analysis module - Version améliorée avec sentiment sur réputation uniquement
"""
import streamlit as st
from typing import Dict, List, Any, Tuple
import re
from collections import Counter
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from translations import get_text
import pandas as pd
from datetime import datetime
import json
import os

# Try to import Vertex AI sentiment analyzer
try:
    from .vertex_sentiment_analyzer import VertexSentimentAnalyzer, create_entity_visualization, \
        create_sentiment_magnitude_scatter

    VERTEX_AVAILABLE = True
except ImportError as e:
    VERTEX_AVAILABLE = False
    print(f"Failed to import Vertex: {e}")

# Try to import optional dependencies
try:
    from textblob import TextBlob

    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    from wordcloud import WordCloud

    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

# Analysis prompts
BRAND_ANALYSIS_PROMPTS = {
    "en": {
        "knowledge": [
            "Tell me everything you know about {brand} company/brand",
            "What is {brand}'s history, mission, and main products or services?",
            "What are {brand}'s key achievements, milestones, and market position?"
        ],
        "reputation": [
            "What do customers and the general public think about {brand}? Include both positive and negative aspects",
            "What are the main customer reviews, testimonials, and feedback about {brand}?",
            "What controversies, issues, or challenges has {brand} faced? How did they handle them?"
        ]
    },
    "fr": {
        "knowledge": [
            "Dis-moi tout ce que tu sais sur l'entreprise/marque {brand}",
            "Quelle est l'histoire, la mission et les principaux produits ou services de {brand} ?",
            "Quels sont les principales réalisations, étapes clés et position sur le marché de {brand} ?"
        ],
        "reputation": [
            "Que pensent les clients et le grand public de {brand} ? Inclure les aspects positifs et négatifs",
            "Quels sont les principaux avis clients, témoignages et retours sur {brand} ?",
            "Quelles controverses, problèmes ou défis {brand} a-t-elle rencontrés ? Comment ont-ils été gérés ?"
        ]
    }
}


def extract_key_themes(text: str, top_n: int = 10) -> List[Tuple[str, int]]:
    """Extract key themes and their frequency from text with focus on sentiment-related terms"""
    lang = st.session_state.get('language', 'en')

    # Enhanced stop words for both languages
    stop_words = {
        # French stop words
        'le', 'la', 'les', 'un', 'une', 'et', 'ou', 'mais', 'dans', 'sur', 'à', 'pour',
        'de', 'du', 'des', 'avec', 'par', 'comme', 'est', 'sont', 'être', 'avoir',
        'ce', 'ces', 'cette', 'cet', 'leur', 'leurs', 'il', 'elle', 'ils', 'elles',
        'je', 'tu', 'nous', 'vous', 'qui', 'que', 'quoi', 'dont', 'où', 'si', 'ne',
        'pas', 'plus', 'très', 'bien', 'tout', 'tous', 'toute', 'toutes', 'aussi',
        'autre', 'autres', 'même', 'faire', 'fait', 'été', 'avoir', 'être', 'peut',
        'peuvent', 'pourrait', 'son', 'ses', 'sa', 'se', 'en', 'y', 'on', 'au', 'aux',
        # English stop words
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
        'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
        'may', 'might', 'must', 'shall', 'can', 'this', 'that', 'these', 'those',
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'them', 'their', 'what', 'which',
        'who', 'when', 'where', 'why', 'how', 'all', 'each', 'every', 'some', 'any',
        'if', 'not', 'no', 'yes', 'so', 'than', 'too', 'very', 'just', 'there', 'here'
    }

    # Generic sentiment and business-related keywords (not specific to any industry)
    priority_keywords = {
        'fr': {
            # Positive
            'qualité', 'excellent', 'parfait', 'rapide', 'facile', 'professionnel',
            'satisfait', 'recommande', 'magnifique', 'pratique', 'efficace', 'fiable',
            'moderne', 'innovant', 'créatif', 'supérieur', 'exceptionnel', 'remarquable',
            # Negative
            'problème', 'erreur', 'lent', 'difficile', 'déçu', 'compliqué',
            'limité', 'défaut', 'manque', 'insuffisant', 'cher', 'coûteux',
            # Neutral/Business
            'service', 'client', 'produit', 'livraison', 'prix', 'commande',
            'entreprise', 'marque', 'expérience', 'achat', 'utilisation'
        },
        'en': {
            # Positive
            'quality', 'excellent', 'perfect', 'fast', 'easy', 'professional',
            'satisfied', 'recommend', 'beautiful', 'practical', 'efficient', 'reliable',
            'modern', 'innovative', 'creative', 'superior', 'exceptional', 'remarkable',
            # Negative
            'problem', 'error', 'slow', 'difficult', 'disappointed', 'complicated',
            'limited', 'defect', 'lack', 'insufficient', 'expensive', 'costly',
            # Neutral/Business
            'service', 'customer', 'product', 'delivery', 'price', 'order',
            'company', 'brand', 'experience', 'purchase', 'usage'
        }
    }

    # Get language-specific priority words
    current_priority = priority_keywords.get(lang, priority_keywords['en'])

    # Extract words with better filtering
    import re
    words = re.findall(r'\b[a-zA-ZÀ-ÿ]{3,}\b', text.lower())

    # Filter and weight words
    word_weights = {}
    for word in words:
        if word not in stop_words and len(word) > 2:
            # Give higher weight to priority keywords
            if word in current_priority:
                word_weights[word] = word_weights.get(word, 0) + 2
            else:
                word_weights[word] = word_weights.get(word, 0) + 1

    # Also extract bi-grams for better context
    bigrams = []
    words_list = text.lower().split()
    for i in range(len(words_list) - 1):
        word1, word2 = words_list[i], words_list[i + 1]
        # Clean words
        word1 = re.sub(r'[^\w\s]', '', word1)
        word2 = re.sub(r'[^\w\s]', '', word2)

        if (word1 not in stop_words and word2 not in stop_words and
                len(word1) > 2 and len(word2) > 2):
            bigram = f"{word1} {word2}"
            # Check if bigram contains priority keywords
            if any(keyword in bigram for keyword in current_priority):
                word_weights[bigram] = word_weights.get(bigram, 0) + 3

    # Convert to sorted list
    sorted_themes = sorted(word_weights.items(), key=lambda x: x[1], reverse=True)

    # Return top themes, ensuring variety
    final_themes = []
    seen_roots = set()

    for theme, count in sorted_themes:
        # Avoid similar words (e.g., "service" and "services")
        root = theme[:5]
        if root not in seen_roots or ' ' in theme:  # Allow bigrams
            final_themes.append((theme, count))
            seen_roots.add(root)
            if len(final_themes) >= top_n:
                break

    return final_themes


def analyze_sentiment(text: str, vertex_analyzer=None) -> Dict[str, Any]:
    """Analyze sentiment using Vertex AI, TextBlob, or fallback"""

    # Priority 1: Try Vertex AI if available
    if vertex_analyzer and VERTEX_AVAILABLE:
        try:
            result = vertex_analyzer.analyze_sentiment(text)
            return result
        except Exception as e:
            st.warning(f"Vertex AI analysis failed: {str(e)}")

    # Priority 2: Try TextBlob if available
    if TEXTBLOB_AVAILABLE:
        try:
            from textblob import TextBlob
            blob = TextBlob(text)

            # Get overall sentiment
            polarity = blob.sentiment.polarity  # -1 to 1
            subjectivity = blob.sentiment.subjectivity  # 0 to 1

            # Classify sentiment
            if polarity > 0.1:
                sentiment = "positive"
            elif polarity < -0.1:
                sentiment = "negative"
            else:
                sentiment = "neutral"

            # Extract sentences with strong sentiment
            positive_sentences = []
            negative_sentences = []

            try:
                for sentence in blob.sentences:
                    sent_polarity = sentence.sentiment.polarity
                    if sent_polarity > 0.3:
                        positive_sentences.append(str(sentence))
                    elif sent_polarity < -0.3:
                        negative_sentences.append(str(sentence))
            except:
                # Simple sentence split if TextBlob sentence parsing fails
                sentences = text.split('. ')
                for sentence in sentences[:10]:
                    sentence_lower = sentence.lower()
                    if any(word in sentence_lower for word in ['excellent', 'great', 'good', 'amazing']):
                        positive_sentences.append(sentence)
                    elif any(word in sentence_lower for word in ['bad', 'poor', 'terrible', 'awful']):
                        negative_sentences.append(sentence)

            return {
                "polarity": polarity,
                "magnitude": abs(polarity),
                "subjectivity": subjectivity,
                "sentiment": sentiment,
                "positive_aspects": positive_sentences[:5],
                "negative_aspects": negative_sentences[:5],
                "api_used": "textblob"
            }

        except Exception as e:
            st.warning(f"TextBlob analysis failed: {str(e)}")

    # Priority 3: Fallback keyword-based analysis
    positive_keywords = ['excellent', 'great', 'good', 'amazing', 'best', 'love', 'perfect', 'wonderful',
                         'fantastic', 'superior', 'outstanding', 'exceptional']
    negative_keywords = ['bad', 'poor', 'worst', 'terrible', 'hate', 'awful', 'disappointing', 'failure',
                         'problem', 'issue', 'concern', 'negative']

    text_lower = text.lower()
    positive_count = sum(1 for word in positive_keywords if word in text_lower)
    negative_count = sum(1 for word in negative_keywords if word in text_lower)

    # Calculate simple polarity
    total_keywords = positive_count + negative_count
    if total_keywords > 0:
        polarity = (positive_count - negative_count) / total_keywords
    else:
        polarity = 0

    # Determine sentiment
    if polarity > 0.1:
        sentiment = "positive"
    elif polarity < -0.1:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    # Extract sample sentences
    sentences = text.split('. ')[:20]
    positive_sentences = []
    negative_sentences = []

    for sentence in sentences:
        sentence_lower = sentence.lower()
        pos_count = sum(1 for word in positive_keywords if word in sentence_lower)
        neg_count = sum(1 for word in negative_keywords if word in sentence_lower)

        if pos_count > neg_count and len(positive_sentences) < 5:
            positive_sentences.append(sentence)
        elif neg_count > pos_count and len(negative_sentences) < 5:
            negative_sentences.append(sentence)

    return {
        "polarity": polarity,
        "magnitude": abs(polarity),
        "subjectivity": 0.5,  # Default neutral subjectivity
        "sentiment": sentiment,
        "positive_aspects": positive_sentences,
        "negative_aspects": negative_sentences,
        "api_used": "fallback"
    }


def verify_sentiment_classification(aspects_list: List[str], expected_sentiment: str, api_client=None,
                                    brand_name: str = "") -> List[str]:
    """Use AI to verify if aspects are correctly classified as positive or negative"""

    if not api_client or not aspects_list:
        return aspects_list

    # Create verification prompt
    aspects_text = "\n".join([f"- {aspect}" for aspect in aspects_list])

    if expected_sentiment == "positive":
        prompt = f"""Analyze the following list of statements about {brand_name if brand_name else 'a brand'} and return ONLY those that are TRULY POSITIVE aspects:

{aspects_text}

Rules:
1. Only include statements that express genuine satisfaction, praise, or positive features
2. EXCLUDE any statement that mentions problems, issues, or negative aspects
3. EXCLUDE neutral or mixed statements
4. If a statement says something like "solved problems" or "despite issues", it's NOT positive
5. Statements like "fast delivery", "excellent quality", "great service" ARE positive
6. Look for words like: excellent, great, perfect, satisfied, recommend, fast, efficient, quality

Return ONLY a JSON array of the truly positive statements, nothing else.
Example: ["Statement 1", "Statement 2"]"""

    else:  # negative
        prompt = f"""Analyze the following list of statements about {brand_name if brand_name else 'a brand'} and return ONLY those that are TRULY NEGATIVE aspects:

{aspects_text}

Rules:
1. Only include statements that express genuine problems, complaints, or dissatisfaction
2. EXCLUDE any statement that praises or compliments the brand
3. EXCLUDE neutral or mixed statements
4. If a statement says "great quality" or "fast delivery", it's NOT negative
5. Statements like "delivery delays", "high prices", "quality issues", "poor service" ARE negative
6. Look for words like: problem, issue, disappointed, slow, expensive, poor, delay, defect

Return ONLY a JSON array of the truly negative statements, nothing else.
Example: ["Statement 1", "Statement 2"]"""

    try:
        # Call API for verification
        if hasattr(api_client, 'client'):  # OpenAI client
            response = api_client.client.chat.completions.create(
                model=api_client.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000
            )
            result = response.choices[0].message.content.strip()
        else:
            result = api_client.call_api(prompt)

        # Parse JSON response
        import json
        try:
            # Clean response
            result = result.strip()
            if result.startswith('```json'):
                result = result[7:]
            if result.startswith('```'):
                result = result[3:]
            if result.endswith('```'):
                result = result[:-3]

            verified_aspects = json.loads(result.strip())

            if isinstance(verified_aspects, list):
                # Return only verified aspects that were in the original list
                return [asp for asp in verified_aspects if asp in aspects_list]

        except json.JSONDecodeError:
            st.warning(f"Failed to parse AI verification response")

    except Exception as e:
        st.warning(f"Verification failed: {str(e)}")

    # Return original list if verification fails
    return aspects_list


def extract_brand_aspects(responses: Dict[str, List[str]], brand_name: str) -> Dict[str, Any]:
    """Extract and categorize brand aspects from LLM responses"""

    aspects = {
        "products_services": [],
        "achievements": [],
        "challenges": [],
        "customer_feedback": [],
        "market_position": [],
        "values_mission": []
    }

    # Keywords for categorization
    aspect_keywords = {
        "products_services": ["product", "service", "offer", "solution", "platform", "produit", "offre"],
        "achievements": ["achievement", "success", "award", "milestone", "growth", "réussite", "succès"],
        "challenges": ["challenge", "issue", "problem", "controversy", "difficulty", "problème", "défi"],
        "customer_feedback": ["customer", "review", "feedback", "testimonial", "user", "client", "avis"],
        "market_position": ["market", "leader", "competitor", "position", "share", "marché", "concurrent"],
        "values_mission": ["mission", "value", "vision", "purpose", "goal", "valeur", "objectif"]
    }

    # Process all responses
    all_text = ""
    for platform_responses in responses.values():
        for response in platform_responses:
            all_text += " " + response

            # Categorize sentences
            sentences = response.split('.')
            for sentence in sentences:
                sentence_lower = sentence.lower()
                for aspect, keywords in aspect_keywords.items():
                    if any(keyword in sentence_lower for keyword in keywords):
                        aspects[aspect].append(sentence.strip())

    # Limit to top entries per aspect
    for aspect in aspects:
        aspects[aspect] = aspects[aspect][:3]

    return aspects, all_text


def create_sentiment_gauge(polarity: float) -> go.Figure:
    """Create a gauge chart for sentiment visualization"""
    lang = st.session_state.get('language', 'en')

    # Convert polarity (-1 to 1) to percentage (0 to 100)
    sentiment_score = (polarity + 1) * 50

    # Determine color based on score
    if sentiment_score >= 70:
        color = "green"
    elif sentiment_score >= 30:
        color = "yellow"
    else:
        color = "red"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=sentiment_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': get_text('sentiment_score', lang)},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 30], 'color': "lightgray"},
                {'range': [30, 70], 'color': "gray"},
                {'range': [70, 100], 'color': "lightgray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))

    fig.update_layout(height=300)
    return fig


def create_word_cloud(text: str, brand_name: str) -> plt.Figure:
    """Create a word cloud from brand-related text with focus on meaningful terms"""
    lang = st.session_state.get('language', 'en')

    if not WORDCLOUD_AVAILABLE:
        # Fallback: create a simple bar chart of top words
        words = extract_key_themes(text, top_n=20)

        fig, ax = plt.subplots(figsize=(10, 5))

        if words:
            word_list, counts = zip(*words)
            # Show only top 15 for readability
            word_list = word_list[:15]
            counts = counts[:15]

            bars = ax.barh(word_list, counts)

            # Color bars based on sentiment
            colors = []
            for word in word_list:
                if any(pos in word for pos in ['quality', 'excellent', 'perfect', 'qualité', 'parfait']):
                    colors.append('#4CAF50')  # Green for positive
                elif any(neg in word for neg in ['problem', 'issue', 'error', 'problème', 'erreur']):
                    colors.append('#F44336')  # Red for negative
                else:
                    colors.append('#2196F3')  # Blue for neutral

            for bar, color in zip(bars, colors):
                bar.set_color(color)

            ax.set_xlabel(get_text('frequency', lang))
            ax.set_title(f"{get_text('key_themes', lang)} - {brand_name}", fontsize=16)
            ax.invert_yaxis()
        else:
            ax.text(0.5, 0.5, 'No significant themes found',
                    ha='center', va='center', transform=ax.transAxes)

        plt.tight_layout()
        return fig

    # Remove brand name to avoid dominance
    text = text.replace(brand_name.lower(), "")
    text = text.replace(brand_name, "")

    # Extract themes for word cloud
    themes = extract_key_themes(text, top_n=100)

    # Create frequency dict for word cloud
    word_freq = {theme[0]: theme[1] for theme in themes}

    # Generate word cloud with custom colors
    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        # Color based on sentiment
        if any(pos in word for pos in ['quality', 'excellent', 'perfect', 'qualité', 'parfait', 'rapide', 'facile']):
            return '#4CAF50'  # Green
        elif any(neg in word for neg in ['problem', 'issue', 'error', 'problème', 'erreur', 'lent', 'difficile']):
            return '#F44336'  # Red
        else:
            return '#2196F3'  # Blue

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        max_words=50,
        relative_scaling=0.5,
        min_font_size=10
    ).generate_from_frequencies(word_freq)

    # Recolor with sentiment colors
    wordcloud.recolor(color_func=color_func)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(f"{get_text('key_themes', lang)} - {brand_name}", fontsize=16)

    return fig


def create_aspect_radar(aspects: Dict[str, List[str]]) -> go.Figure:
    """Create radar chart showing brand aspect coverage"""

    categories = list(aspects.keys())
    values = [len(v) for v in aspects.values()]

    # Normalize values to 0-100 scale
    max_val = max(values) if values else 1
    normalized_values = [(v / max_val) * 100 for v in values]

    fig = go.Figure(data=go.Scatterpolar(
        r=normalized_values,
        theta=[cat.replace('_', ' ').title() for cat in categories],
        fill='toself',
        name='Brand Coverage'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=False,
        title="Brand Aspect Coverage in LLM Responses"
    )

    return fig


def run_brand_llm_analysis(
        brand_name: str,
        api_clients: Dict[str, Any],
        selected_models: Dict[str, str],
        vertex_credentials_path: str = None
) -> Dict[str, Any]:
    """Run comprehensive brand analysis using LLMs"""

    lang = st.session_state.get('language', 'en')
    prompts = BRAND_ANALYSIS_PROMPTS[lang]

    # Initialize Vertex AI analyzer if available
    vertex_analyzer = None
    if VERTEX_AVAILABLE and vertex_credentials_path:
        try:
            vertex_analyzer = VertexSentimentAnalyzer(vertex_credentials_path)
            if vertex_analyzer.available:
                st.success("✅ Using Google Cloud Natural Language API for sentiment analysis")
            else:
                st.info("ℹ️ Google Cloud NL API not configured, using alternative methods")
                vertex_analyzer = None
        except Exception as e:
            st.warning(f"Failed to initialize Vertex AI: {str(e)}")
            vertex_analyzer = None

    # Initialize results
    results = {
        "knowledge_responses": {},
        "reputation_responses": {},
        "sentiment_analysis": {},
        "brand_aspects": {},
        "key_themes": [],
        "entities": [],
        "api_clients": api_clients  # Store for verification later
    }

    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()

    total_steps = len(api_clients) * (len(prompts["knowledge"]) + len(prompts["reputation"]))
    current_step = 0

    # Phase 1: Knowledge Analysis
    status_text.text(get_text('analyzing_brand_knowledge', lang))
    for platform_key, client in api_clients.items():
        results["knowledge_responses"][platform_key] = []

        for prompt in prompts["knowledge"]:
            formatted_prompt = prompt.format(brand=brand_name)

            if platform_key == 'perplexity':
                response, _ = client.call_api(formatted_prompt)
            else:
                response = client.call_api(formatted_prompt)

            results["knowledge_responses"][platform_key].append(response)

            current_step += 1
            progress_bar.progress(current_step / total_steps)

    # Phase 2: Reputation Analysis
    status_text.text(get_text('analyzing_brand_reputation', lang))
    for platform_key, client in api_clients.items():
        results["reputation_responses"][platform_key] = []

        for prompt in prompts["reputation"]:
            formatted_prompt = prompt.format(brand=brand_name)

            if platform_key == 'perplexity':
                response, _ = client.call_api(formatted_prompt)
            else:
                response = client.call_api(formatted_prompt)

            results["reputation_responses"][platform_key].append(response)

            current_step += 1
            progress_bar.progress(current_step / total_steps)

    # Phase 3: Sentiment Analysis - UNIQUEMENT SUR REPUTATION
    status_text.text(get_text('processing_sentiment', lang))

    # Sentiment analysis per platform - REPUTATION ONLY
    for platform in api_clients.keys():
        # Analyser UNIQUEMENT les réponses de réputation
        reputation_text = " ".join(results["reputation_responses"][platform])
        results["sentiment_analysis"][platform] = analyze_sentiment(reputation_text, vertex_analyzer)

    # Overall sentiment - REPUTATION ONLY
    all_reputation_responses = []
    for responses in results["reputation_responses"].values():
        all_reputation_responses.extend(responses)

    combined_reputation_text = " ".join(all_reputation_responses)
    results["overall_sentiment"] = analyze_sentiment(combined_reputation_text, vertex_analyzer)

    # Extract entities using Vertex AI if available - Sur tout le contenu
    if vertex_analyzer and VERTEX_AVAILABLE:
        try:
            # Combiner knowledge ET reputation pour l'extraction d'entités
            all_responses = []
            for responses in results["knowledge_responses"].values():
                all_responses.extend(responses)
            for responses in results["reputation_responses"].values():
                all_responses.extend(responses)

            combined_all_text = " ".join(all_responses)
            results["entities"] = vertex_analyzer.analyze_entities(combined_all_text[:5000])
        except Exception as e:
            results["entities"] = []

    # Extract brand aspects - Sur tout le contenu (knowledge + reputation)
    all_platform_responses = {}
    for platform in api_clients.keys():
        all_platform_responses[platform] = (results["knowledge_responses"][platform] +
                                            results["reputation_responses"][platform])

    results["brand_aspects"], full_text = extract_brand_aspects(all_platform_responses, brand_name)

    # Extract key themes - UNIQUEMENT de la réputation
    reputation_only_text = " ".join(all_reputation_responses)  # Déjà créé plus haut
    results["key_themes"] = extract_key_themes(reputation_only_text, top_n=15)

    # Clear progress
    progress_bar.empty()
    status_text.empty()

    return results


def display_brand_llm_analysis(analysis_results: Dict[str, Any], brand_name: str):
    """Display brand LLM analysis results"""

    lang = st.session_state.get('language', 'en')

    st.header(f"🔍 {brand_name} - {get_text('brand_llm_analysis', lang)}")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    overall_sentiment = analysis_results["overall_sentiment"]

    with col1:
        sentiment_emoji = {"positive": "😊", "negative": "😟", "neutral": "😐"}
        st.metric(
            get_text('overall_sentiment', lang),
            sentiment_emoji[overall_sentiment["sentiment"]],
            f"{overall_sentiment['polarity']:.2f}"
        )

    with col2:
        st.metric(
            get_text('subjectivity_score', lang),
            f"{overall_sentiment['subjectivity']:.2f}",
            help="0 = Objective, 1 = Subjective" if lang == 'en' else "0 = Objectif, 1 = Subjectif"
        )

    with col3:
        total_themes = len(analysis_results["key_themes"])
        st.metric(get_text('key_themes', lang), total_themes)

    with col4:
        aspects_covered = sum(1 for v in analysis_results["brand_aspects"].values() if v)
        st.metric(get_text('aspects_covered', lang), f"{aspects_covered}/6")

    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        f"📊 {get_text('sentiment_analysis_tab', lang)}",
        f"☁️ {get_text('key_themes_tab', lang)}",
        f"🎯 {get_text('brand_aspects_tab', lang)}",
        f"📝 {get_text('knowledge_summary_tab', lang)}",
        f"⭐ {get_text('reputation_summary_tab', lang)}"
    ])

    with tab1:
        # Sentiment gauge
        st.subheader(get_text('overall_brand_sentiment', lang))
        st.info("ℹ️ " + ("Sentiment analysis is based on reputation responses only" if lang == 'en'
                         else "L'analyse de sentiment est basée uniquement sur les réponses de réputation"))
        fig_gauge = create_sentiment_gauge(overall_sentiment["polarity"])
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Additional metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            polarity = overall_sentiment.get('polarity', 0)
            st.metric(
                "Polarity Score" if lang == 'en' else "Score de polarité",
                f"{polarity:.3f}",
                delta=None if abs(polarity) < 0.05 else (
                    "Positive" if polarity > 0 else "Negative") if lang == 'en' else (
                    "Positif" if polarity > 0 else "Négatif"),
                help="Score from -1 (very negative) to +1 (very positive)" if lang == 'en' else "Score de -1 (très négatif) à +1 (très positif)"
            )

        with col2:
            magnitude = overall_sentiment.get('magnitude', 0)
            st.metric(
                "Magnitude",
                f"{magnitude:.1f}",
                help="Sentiment strength (0 = weak, 10+ = strong)" if lang == 'en' else "Force du sentiment (0 = faible, 10+ = fort)"
            )

        with col3:
            pos_count = overall_sentiment.get('positive_count', len(overall_sentiment.get('positive_aspects', [])))
            st.metric(
                "Positive Aspects" if lang == 'en' else "Aspects positifs",
                pos_count,
                help="Total number of positive sentences detected" if lang == 'en' else "Nombre total de phrases positives détectées"
            )

        with col4:
            neg_count = overall_sentiment.get('negative_count', len(overall_sentiment.get('negative_aspects', [])))
            st.metric(
                "Negative Aspects" if lang == 'en' else "Aspects négatifs",
                neg_count,
                help="Total number of negative sentences detected" if lang == 'en' else "Nombre total de phrases négatives détectées"
            )

        # Platform comparison
        st.subheader(get_text('sentiment_by_platform', lang))

        platforms = []
        polarities = []
        pos_counts = []
        neg_counts = []
        colors = []

        for platform, sentiment_data in analysis_results["sentiment_analysis"].items():
            platforms.append(platform.title())
            polarity = sentiment_data.get('polarity', 0)
            polarities.append(polarity)
            pos_counts.append(sentiment_data.get('positive_count', len(sentiment_data.get('positive_aspects', []))))
            neg_counts.append(sentiment_data.get('negative_count', len(sentiment_data.get('negative_aspects', []))))

            if polarity > 0.05:
                colors.append('#4CAF50')  # Green
            elif polarity < -0.05:
                colors.append('#F44336')  # Red
            else:
                colors.append('#FFC107')  # Yellow

        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                'Sentiment Score by Platform' if lang == 'en' else 'Score de Sentiment par Plateforme',
                'Positive vs Negative Aspects Count' if lang == 'en' else 'Nombre d\'Aspects Positifs vs Négatifs'
            ),
            column_widths=[0.5, 0.5]
        )

        # Chart 1: Polarity score
        fig.add_trace(
            go.Bar(
                x=platforms,
                y=polarities,
                marker_color=colors,
                text=[f"{p:.3f}" for p in polarities],
                textposition='outside',
                name='Polarity' if lang == 'en' else 'Polarité',
                showlegend=False
            ),
            row=1, col=1
        )

        # Chart 2: Positive vs negative aspects
        fig.add_trace(
            go.Bar(
                x=platforms,
                y=pos_counts,
                name='Positive Aspects' if lang == 'en' else 'Aspects Positifs',
                marker_color='#4CAF50',
                text=pos_counts,
                textposition='outside'
            ),
            row=1, col=2
        )

        fig.add_trace(
            go.Bar(
                x=platforms,
                y=neg_counts,
                name='Negative Aspects' if lang == 'en' else 'Aspects Négatifs',
                marker_color='#F44336',
                text=neg_counts,
                textposition='outside'
            ),
            row=1, col=2
        )

        fig.update_xaxes(tickangle=-45)
        fig.update_yaxes(range=[-0.5, 0.5], row=1, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)

        fig.update_layout(
            height=400,
            showlegend=True,
            barmode='group'
        )

        st.plotly_chart(fig, use_container_width=True)

        # Summary table
        st.subheader("📊 " + ("Detailed Summary by Platform" if lang == 'en' else "Résumé détaillé par plateforme"))

        summary_data = []
        for platform, sentiment_data in analysis_results["sentiment_analysis"].items():
            summary_data.append({
                "Platform" if lang == 'en' else "Plateforme": platform.title(),
                "Polarity" if lang == 'en' else "Polarité": f"{sentiment_data.get('polarity', 0):.3f}",
                "Magnitude": f"{sentiment_data.get('magnitude', 0):.1f}",
                "Positive Aspects" if lang == 'en' else "Aspects Positifs": sentiment_data.get('positive_count',
                                                                                               len(sentiment_data.get(
                                                                                                   'positive_aspects',
                                                                                                   []))),
                "Negative Aspects" if lang == 'en' else "Aspects Négatifs": sentiment_data.get('negative_count',
                                                                                               len(sentiment_data.get(
                                                                                                   'negative_aspects',
                                                                                                   []))),
                "API Used" if lang == 'en' else "API utilisée": sentiment_data.get('api_used', 'unknown')
            })

        df_summary = pd.DataFrame(summary_data)
        st.dataframe(df_summary, use_container_width=True, hide_index=True)

        # Positive and negative aspects - UNIQUEMENT DE LA REPUTATION
        col1, col2 = st.columns(2)

        # Get an API client for verification (prefer OpenAI/ChatGPT)
        verification_client = None
        if 'chatgpt' in analysis_results.get('api_clients', {}):
            verification_client = analysis_results['api_clients']['chatgpt']
        elif 'gemini' in analysis_results.get('api_clients', {}):
            verification_client = analysis_results['api_clients']['gemini']

        with col1:
            st.subheader(f"✅ {get_text('positive_aspects', lang)}")

            # Collecter UNIQUEMENT les aspects positifs de l'analyse de sentiment
            all_positive = []
            for platform_data in analysis_results["sentiment_analysis"].values():
                positive_aspects = platform_data.get("positive_aspects", [])
                for aspect in positive_aspects:
                    if aspect and len(aspect) > 20 and aspect not in all_positive:
                        all_positive.append(aspect)

            # VERIFICATION PAR AI si disponible
            if verification_client and all_positive:
                with st.spinner(
                        "Verifying positive aspects..." if lang == 'en' else "Vérification des aspects positifs..."):
                    all_positive = verify_sentiment_classification(
                        all_positive,
                        "positive",
                        verification_client,
                        brand_name
                    )

            # Afficher
            if all_positive:
                for i, aspect in enumerate(all_positive[:8]):
                    st.write(f"• {aspect}")
            else:
                st.info("No specific positive aspects detected in reputation analysis." if lang == 'en'
                        else "Aucun aspect positif spécifique détecté dans l'analyse de réputation.")

        with col2:
            st.subheader(f"❌ {get_text('negative_aspects', lang)}")

            # Collecter UNIQUEMENT les aspects négatifs de l'analyse de sentiment
            all_negative = []
            for platform_data in analysis_results["sentiment_analysis"].values():
                negative_aspects = platform_data.get("negative_aspects", [])
                for aspect in negative_aspects:
                    if aspect and len(aspect) > 20 and aspect not in all_negative:
                        all_negative.append(aspect)

            # Si pas assez d'aspects négatifs trouvés, chercher dans les réponses
            if len(all_negative) < 3:
                # Generic negative keywords (not industry-specific)
                negative_keywords = [
                    'problem', 'problems', 'issue', 'issues', 'expensive', 'costly', 'high price',
                    'slow', 'delay', 'delayed', 'difficult', 'complicated', 'disappointed',
                    'poor', 'bad', 'worse', 'worst', 'fail', 'failure', 'error', 'defect',
                    'complaint', 'complain', 'negative', 'dissatisfied', 'unhappy',
                    'problème', 'problèmes', 'cher', 'coûteux', 'prix élevé',
                    'lent', 'retard', 'difficile', 'compliqué', 'déçu', 'décevant',
                    'mauvais', 'pire', 'échec', 'erreur', 'défaut',
                    'plainte', 'négatif', 'insatisfait', 'mécontent'
                ]

                # Context that confirms positive sentiment (even with negative words)
                positive_context = [
                    'highlights the quality', 'emphasize the speed', 'perfect',
                    'appreciate', 'satisfied', 'happy', 'love', 'recommend',
                    'excellent', 'fast', 'reliable', 'efficient', 'great',
                    'souligne la qualité', 'soulignent la rapidité', 'parfait',
                    'apprécient', 'satisfait', 'heureux', 'aime', 'recommande',
                    'excellent', 'rapide', 'fiable', 'efficace'
                ]

                for platform_responses in analysis_results.get("reputation_responses", {}).values():
                    for response in platform_responses:
                        sentences = response.split('. ')
                        for sent in sentences:
                            sent = sent.strip()
                            if len(sent) > 30:
                                sent_lower = sent.lower()

                                # Skip if contains positive expressions
                                if any(pos_expr in sent_lower for pos_expr in positive_context):
                                    continue

                                # Patterns that indicate real problems
                                real_negative_patterns = [
                                    'customers complain', 'users report', 'people mention',
                                    'issues with', 'problems with', 'complaints about',
                                    'delivery delays', 'shipping delays', 'delayed delivery',
                                    'high price', 'expensive', 'overpriced',
                                    'poor quality', 'quality issues', 'defective',
                                    'bad service', 'poor service', 'terrible service',
                                    'clients se plaignent', 'utilisateurs signalent',
                                    'problèmes avec', 'plaintes concernant',
                                    'retards de livraison', 'prix élevé', 'trop cher',
                                    'mauvaise qualité', 'problèmes de qualité',
                                    'mauvais service', 'service médiocre'
                                ]

                                # Check if really negative
                                is_really_negative = any(pattern in sent_lower for pattern in real_negative_patterns)

                                if is_really_negative:
                                    # Final check: no positive contradictions
                                    exclude_patterns = [
                                        'no problem', 'no issue', 'resolved', 'solved',
                                        'improved', 'fixed', 'despite', 'although',
                                        'pas de problème', 'aucun problème', 'résolu',
                                        'amélioré', 'corrigé', 'malgré'
                                    ]
                                    if not any(pattern in sent_lower for pattern in exclude_patterns):
                                        if sent not in all_negative and len(all_negative) < 8:
                                            all_negative.append(sent)

            # VERIFICATION PAR AI si disponible
            if verification_client and all_negative:
                with st.spinner(
                        "Verifying negative aspects..." if lang == 'en' else "Vérification des aspects négatifs..."):
                    all_negative = verify_sentiment_classification(
                        all_negative,
                        "negative",
                        verification_client,
                        brand_name
                    )

            # Afficher
            if all_negative:
                for i, aspect in enumerate(all_negative[:8]):
                    st.write(f"• {aspect}")
            else:
                st.info("No significant negative aspects detected in reputation analysis." if lang == 'en'
                        else "Aucun aspect négatif significatif détecté dans l'analyse de réputation.")

    with tab2:
        # Word cloud - Based ONLY on reputation
        st.subheader(get_text('brand_theme_cloud', lang))
        st.info("ℹ️ " + ("Themes are extracted from reputation responses only" if lang == 'en'
                         else "Les thèmes sont extraits uniquement des réponses de réputation"))

        # Use only reputation texts
        reputation_text = " ".join(
            [r for responses in analysis_results["reputation_responses"].values() for r in responses])

        fig_cloud = create_word_cloud(reputation_text, brand_name)
        st.pyplot(fig_cloud)

        # Top themes table
        st.subheader(get_text('top_themes', lang))
        themes_data = [{get_text('theme', lang): theme[0], get_text('frequency', lang): theme[1]}
                       for theme in analysis_results["key_themes"]]
        df_themes = pd.DataFrame(themes_data)

        fig_themes = px.bar(
            df_themes,
            x=get_text('frequency', lang),
            y=get_text('theme', lang),
            orientation="h",
            title=get_text('most_frequent_themes', lang)
        )
        fig_themes.update_layout(height=600)
        st.plotly_chart(fig_themes, use_container_width=True)

    with tab3:
        # Brand aspects radar
        st.subheader(get_text('brand_aspect_coverage', lang))
        fig_radar = create_aspect_radar(analysis_results["brand_aspects"])
        st.plotly_chart(fig_radar, use_container_width=True)

        # Detailed aspects
        st.subheader(get_text('detailed_brand_aspects', lang))
        for aspect, mentions in analysis_results["brand_aspects"].items():
            if mentions:
                with st.expander(f"📌 {aspect.replace('_', ' ').title()}"):
                    for mention in mentions:
                        st.write(f"• {mention}")

    with tab4:
        st.subheader(f"📚 {get_text('what_llms_know', lang)} {brand_name}")

        for platform, responses in analysis_results["knowledge_responses"].items():
            with st.expander(f"{platform.title()} {get_text('knowledge_summary_tab', lang)}"):
                for i, response in enumerate(responses, 1):
                    st.markdown(f"**{get_text('query', lang)} {i}:**")
                    st.write(response)
                    st.divider()

    with tab5:
        st.subheader(f"⭐ {get_text('brand_reputation_analysis', lang)}")

        for platform, responses in analysis_results["reputation_responses"].items():
            with st.expander(f"{platform.title()} {get_text('reputation_summary_tab', lang)}"):
                for i, response in enumerate(responses, 1):
                    st.markdown(f"**{get_text('query', lang)} {i}:**")
                    st.write(response)
                    st.divider()

    # Export button
    if st.button(f"📥 {get_text('export_brand_analysis', lang)}"):
        export_data = {
            "brand_name": brand_name,
            "analysis_date": datetime.now().isoformat(),
            "overall_sentiment": overall_sentiment,
            "platform_sentiments": analysis_results["sentiment_analysis"],
            "key_themes": analysis_results["key_themes"],
            "brand_aspects": analysis_results["brand_aspects"]
        }

        json_str = json.dumps(export_data, indent=2)
        st.download_button(
            label=get_text('download_brand_analysis', lang),
            data=json_str,
            file_name=f"{brand_name}_brand_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )