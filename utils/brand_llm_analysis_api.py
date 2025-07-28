"""
Brand LLM Analysis module - API version without Streamlit dependencies
"""
from typing import Dict, List, Any, Tuple
import re
from collections import Counter
import os

# Try to import optional dependencies
try:
    from textblob import TextBlob

    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

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
    """Extract key themes and their frequency from text"""
    # Enhanced stop words for both languages
    stop_words = {
        'le', 'la', 'les', 'un', 'une', 'et', 'ou', 'mais', 'dans', 'sur', 'à', 'pour',
        'de', 'du', 'des', 'avec', 'par', 'comme', 'est', 'sont', 'être', 'avoir',
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
        'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be', 'been', 'being',
    }

    # Extract words
    words = re.findall(r'\b[a-zA-ZÀ-ÿ]{3,}\b', text.lower())

    # Filter and count
    word_counts = Counter()
    for word in words:
        if word not in stop_words and len(word) > 2:
            word_counts[word] += 1

    return word_counts.most_common(top_n)


def analyze_sentiment(text: str, vertex_analyzer=None) -> Dict[str, Any]:
    """Analyze sentiment using available methods"""

    # Try Vertex AI if available
    if vertex_analyzer:
        try:
            return vertex_analyzer.analyze_sentiment(text)
        except:
            pass

    # Fallback to TextBlob if available
    if TEXTBLOB_AVAILABLE:
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity

            if polarity > 0.1:
                sentiment = "positive"
            elif polarity < -0.1:
                sentiment = "negative"
            else:
                sentiment = "neutral"

            return {
                "polarity": polarity,
                "magnitude": abs(polarity),
                "subjectivity": subjectivity,
                "sentiment": sentiment,
                "positive_aspects": [],
                "negative_aspects": [],
                "api_used": "textblob"
            }
        except:
            pass

    # Simple fallback
    positive_keywords = ['excellent', 'great', 'good', 'amazing', 'best', 'love']
    negative_keywords = ['bad', 'poor', 'worst', 'terrible', 'hate', 'awful']

    text_lower = text.lower()
    positive_count = sum(1 for word in positive_keywords if word in text_lower)
    negative_count = sum(1 for word in negative_keywords if word in text_lower)

    total = positive_count + negative_count
    polarity = (positive_count - negative_count) / total if total > 0 else 0

    return {
        "polarity": polarity,
        "magnitude": abs(polarity),
        "subjectivity": 0.5,
        "sentiment": "positive" if polarity > 0.1 else "negative" if polarity < -0.1 else "neutral",
        "positive_aspects": [],
        "negative_aspects": [],
        "positive_count": positive_count,
        "negative_count": negative_count,
        "api_used": "fallback"
    }


def extract_brand_aspects(responses: Dict[str, List[str]], brand_name: str) -> Tuple[Dict[str, List[str]], str]:
    """Extract and categorize brand aspects from LLM responses"""

    aspects = {
        "products_services": [],
        "achievements": [],
        "challenges": [],
        "customer_feedback": [],
        "market_position": [],
        "values_mission": []
    }

    aspect_keywords = {
        "products_services": ["product", "service", "offer", "solution", "platform"],
        "achievements": ["achievement", "success", "award", "milestone", "growth"],
        "challenges": ["challenge", "issue", "problem", "controversy", "difficulty"],
        "customer_feedback": ["customer", "review", "feedback", "testimonial", "user"],
        "market_position": ["market", "leader", "competitor", "position", "share"],
        "values_mission": ["mission", "value", "vision", "purpose", "goal"]
    }

    all_text = ""
    for platform_responses in responses.values():
        for response in platform_responses:
            all_text += " " + response

            sentences = response.split('.')
            for sentence in sentences:
                sentence_lower = sentence.lower()
                for aspect, keywords in aspect_keywords.items():
                    if any(keyword in sentence_lower for keyword in keywords):
                        aspects[aspect].append(sentence.strip())

    # Limit to top entries
    for aspect in aspects:
        aspects[aspect] = aspects[aspect][:3]

    return aspects, all_text


def run_brand_llm_analysis(
        brand_name: str,
        api_clients: Dict[str, Any],
        selected_models: Dict[str, str],
        vertex_credentials_path: str = None
) -> Dict[str, Any]:
    """Run comprehensive brand analysis using LLMs"""

    lang = 'fr'  # Default to French for API
    prompts = BRAND_ANALYSIS_PROMPTS[lang]

    # Initialize Vertex AI analyzer if available
    vertex_analyzer = None
    if vertex_credentials_path and os.path.exists(vertex_credentials_path):
        try:
            from utils.vertex_sentiment_analyzer import VertexSentimentAnalyzer
            vertex_analyzer = VertexSentimentAnalyzer(vertex_credentials_path)
        except:
            pass

    # Initialize results
    results = {
        "knowledge_responses": {},
        "reputation_responses": {},
        "sentiment_analysis": {},
        "brand_aspects": {},
        "key_themes": [],
        "entities": []
    }

    # Phase 1: Knowledge Analysis
    for platform_key, client in api_clients.items():
        results["knowledge_responses"][platform_key] = []

        for prompt in prompts["knowledge"]:
            formatted_prompt = prompt.format(brand=brand_name)

            try:
                if platform_key == 'perplexity':
                    response, _ = client.call_api(formatted_prompt)
                else:
                    response = client.call_api(formatted_prompt)

                results["knowledge_responses"][platform_key].append(response)
            except Exception as e:
                results["knowledge_responses"][platform_key].append(f"Error: {str(e)}")

    # Phase 2: Reputation Analysis
    for platform_key, client in api_clients.items():
        results["reputation_responses"][platform_key] = []

        for prompt in prompts["reputation"]:
            formatted_prompt = prompt.format(brand=brand_name)

            try:
                if platform_key == 'perplexity':
                    response, _ = client.call_api(formatted_prompt)
                else:
                    response = client.call_api(formatted_prompt)

                results["reputation_responses"][platform_key].append(response)
            except Exception as e:
                results["reputation_responses"][platform_key].append(f"Error: {str(e)}")

    # Phase 3: Sentiment Analysis on reputation only
    for platform in api_clients.keys():
        reputation_text = " ".join(results["reputation_responses"][platform])
        results["sentiment_analysis"][platform] = analyze_sentiment(reputation_text, vertex_analyzer)

    # Overall sentiment
    all_reputation_responses = []
    for responses in results["reputation_responses"].values():
        all_reputation_responses.extend(responses)

    combined_reputation_text = " ".join(all_reputation_responses)
    results["overall_sentiment"] = analyze_sentiment(combined_reputation_text, vertex_analyzer)

    # Extract brand aspects
    all_platform_responses = {}
    for platform in api_clients.keys():
        all_platform_responses[platform] = (results["knowledge_responses"][platform] +
                                            results["reputation_responses"][platform])

    results["brand_aspects"], full_text = extract_brand_aspects(all_platform_responses, brand_name)

    # Extract key themes from reputation
    results["key_themes"] = extract_key_themes(combined_reputation_text, top_n=15)

    return results