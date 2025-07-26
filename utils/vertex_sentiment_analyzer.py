"""
Google Vertex AI Sentiment Analysis module - Version avec comptage correct et filtrage amélioré
"""
from google.cloud import language_v1
from google.oauth2 import service_account
import streamlit as st
from typing import Dict, List, Any, Tuple
import os
import json
import plotly.graph_objects as go
import re


class VertexSentimentAnalyzer:
    """Sentiment analyzer using Google Cloud Natural Language API"""

    def __init__(self, credentials_path: str = None):
        """
        Initialize the Vertex AI client

        Args:
            credentials_path: Path to service account JSON file
        """
        self.available = False
        self.client = None
        self.error_message = None

        try:
            if credentials_path and os.path.exists(credentials_path):
                print(f"=== VERTEX INIT ===")
                print(f"• Loading credentials from: {credentials_path}")

                credentials = service_account.Credentials.from_service_account_file(
                    credentials_path
                )
                self.client = language_v1.LanguageServiceClient(credentials=credentials)
                self.available = True
                print(f"✅ Vertex client created successfully")
            else:
                # Try default credentials
                print(f"• No credentials path provided, trying default")
                self.client = language_v1.LanguageServiceClient()
                self.available = True
        except Exception as e:
            self.error_message = str(e)
            print(f"❌ Vertex init failed: {self.error_message}")
            st.warning(f"Google Cloud Natural Language API not configured: {str(e)}")
            self.available = False
            self.client = None

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment using Google Cloud Natural Language API

        Args:
            text: Text to analyze

        Returns:
            Dictionary with sentiment analysis results including scores for debug
        """
        print(f"\n=== VERTEX ANALYZER: analyze_sentiment called ===")
        print(f"• Available: {self.available}")
        print(f"• Client exists: {self.client is not None}")
        print(f"• Text length: {len(text)}")
        print(f"• Text preview: {text[:100]}...")

        if not self.available or not self.client:
            print(f"• Falling back - available={self.available}, client={self.client is not None}")
            return self._fallback_sentiment_analysis(text)

        try:
            # Limiter le texte
            text_to_analyze = text[:5000] if len(text) > 5000 else text
            print(f"• Analyzing {len(text_to_analyze)} characters")

            # Prepare the document
            document = language_v1.Document(
                content=text_to_analyze,
                type_=language_v1.Document.Type.PLAIN_TEXT,
                language="en"  # Auto-detect if needed
            )

            print("• Calling Google Cloud Natural Language API...")

            # Analyze sentiment
            sentiment_response = self.client.analyze_sentiment(
                request={'document': document}
            )

            # Get overall sentiment
            sentiment = sentiment_response.document_sentiment

            print(f"✅ API Response received:")
            print(f"  - Score: {sentiment.score}")
            print(f"  - Magnitude: {sentiment.magnitude}")
            print(f"  - Sentences analyzed: {len(sentiment_response.sentences)}")

            # Classify sentiment
            if sentiment.score > 0.25:
                sentiment_label = "positive"
            elif sentiment.score < -0.25:
                sentiment_label = "negative"
            else:
                sentiment_label = "neutral"

            # Extract sentence-level sentiments WITH SCORES
            positive_sentences = []
            negative_sentences = []
            positive_debug = []  # Pour debug avec scores
            negative_debug = []  # Pour debug avec scores

            for i, sentence in enumerate(sentiment_response.sentences):
                sentence_text = sentence.text.content.strip()
                sentence_score = sentence.sentiment.score

                if i < 5:  # Debug les 5 premières phrases
                    print(f"  - Sentence {i}: score={sentence_score:.3f}, text={sentence_text[:50]}...")

                # Skip very short sentences
                if len(sentence_text) < 20:
                    continue

                # IMPORTANT: Stocker avec le score pour debug
                sentence_info = {
                    "text": sentence_text,
                    "score": sentence_score
                }

                # Use sentence-specific sentiment score with STRICTER thresholds
                if sentence_score > 0.3:  # Plus strict pour éviter les faux positifs
                    positive_sentences.append(sentence_text)
                    positive_debug.append(sentence_info)
                elif sentence_score < -0.3:  # Plus strict pour éviter les faux négatifs
                    negative_sentences.append(sentence_text)
                    negative_debug.append(sentence_info)

            # Log pour vérifier
            print(f"\n=== VERIFICATION DES CLASSIFICATIONS ===")
            print(f"POSITIVES ({len(positive_sentences)}):")
            for i, info in enumerate(positive_debug[:3]):
                print(f"  {i + 1}. Score={info['score']:.3f}: {info['text'][:60]}...")

            print(f"\nNEGATIVES ({len(negative_sentences)}):")
            for i, info in enumerate(negative_debug[:3]):
                print(f"  {i + 1}. Score={info['score']:.3f}: {info['text'][:60]}...")

            print(f"\n• Positive sentences found: {len(positive_sentences)}")
            print(f"• Negative sentences found: {len(negative_sentences)}")

            # If not enough sentences found, supplement with manual analysis
            if len(positive_sentences) < 2 or len(negative_sentences) < 2:
                print("• Supplementing with manual analysis...")
                self._supplement_sentences(text, positive_sentences, negative_sentences)
                print(f"• After supplement - Positive: {len(positive_sentences)}, Negative: {len(negative_sentences)}")

            # Stocker les résultats
            result = {
                "polarity": float(sentiment.score),
                "magnitude": float(sentiment.magnitude),
                "subjectivity": min(sentiment.magnitude / 5.0, 1.0),
                "sentiment": sentiment_label,
                "positive_aspects": positive_sentences[:5],  # Exemples limités à 5
                "negative_aspects": negative_sentences[:5],  # Exemples limités à 5
                "positive_count": len(positive_sentences),  # COMPTE TOTAL RÉEL
                "negative_count": len(negative_sentences),  # COMPTE TOTAL RÉEL
                "positive_debug": positive_debug[:5],  # DEBUG: avec scores
                "negative_debug": negative_debug[:5],  # DEBUG: avec scores
                "api_used": "google_cloud_nl"
            }

            print(f"✅ Returning result with api_used={result['api_used']}")
            print(
                f"   Counts in result: positive_count={result['positive_count']}, negative_count={result['negative_count']}")
            return result

        except Exception as e:
            error_str = str(e)
            print(f"❌ Google Cloud NL API error: {error_str}")
            print(f"• Exception type: {type(e).__name__}")
            st.warning(f"Error using Google Cloud NL API: {str(e)}")
            return self._fallback_sentiment_analysis(text)

    def analyze_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract entities from text using Google Cloud Natural Language API

        Args:
            text: Text to analyze

        Returns:
            List of entities with their types and salience
        """
        if not self.available or not self.client:
            return []

        try:
            document = language_v1.Document(
                content=text,
                type_=language_v1.Document.Type.PLAIN_TEXT,
            )

            # Analyze entities
            entities_response = self.client.analyze_entities(
                request={'document': document}
            )

            entities = []
            for entity in entities_response.entities:
                entities.append({
                    "name": entity.name,
                    "type": entity.type_.name,
                    "salience": entity.salience,
                    "mentions": len(entity.mentions),
                    "wikipedia_url": entity.metadata.get("wikipedia_url", "")
                })

            # Sort by salience
            entities.sort(key=lambda x: x["salience"], reverse=True)

            return entities[:10]  # Top 10 entities

        except Exception as e:
            st.warning(f"Error analyzing entities: {str(e)}")
            return []

    def analyze_syntax(self, text: str) -> Dict[str, Any]:
        """
        Analyze syntax and extract key phrases

        Args:
            text: Text to analyze

        Returns:
            Dictionary with syntax analysis results
        """
        if not self.available or not self.client:
            return {"key_phrases": [], "pos_tags": {}}

        try:
            document = language_v1.Document(
                content=text[:1000],  # Limit text length for syntax analysis
                type_=language_v1.Document.Type.PLAIN_TEXT,
            )

            # Analyze syntax
            syntax_response = self.client.analyze_syntax(
                request={'document': document}
            )

            # Extract key phrases (noun phrases)
            key_phrases = []
            pos_tags = {}

            for token in syntax_response.tokens:
                # Count POS tags
                pos = token.part_of_speech.tag.name
                pos_tags[pos] = pos_tags.get(pos, 0) + 1

                # Extract noun phrases
                if pos in ["NOUN", "PROPER_NOUN"] and token.dependency_edge.label.name == "NSUBJ":
                    key_phrases.append(token.text.content)

            return {
                "key_phrases": key_phrases[:10],
                "pos_tags": pos_tags
            }

        except Exception as e:
            return {"key_phrases": [], "pos_tags": {}}

    def _supplement_sentences(self, text: str, positive_sentences: List[str], negative_sentences: List[str]):
        """
        Supplement sentence lists with manual analysis if needed
        WITH BETTER FILTERING to avoid misclassification

        Args:
            text: Original text
            positive_sentences: List to append positive sentences to
            negative_sentences: List to append negative sentences to
        """
        # Manual sentence splitting
        manual_sentences = re.split(r'[.!?]+', text)

        # Keywords for classification - EXPANDED and REFINED
        positive_keywords = {
            'en': ['excellent', 'great', 'good', 'amazing', 'love', 'perfect',
                   'wonderful', 'fantastic', 'best', 'happy', 'satisfied',
                   'quality', 'recommend', 'reliable', 'efficient', 'beautiful',
                   'impressive', 'outstanding', 'awesome', 'brilliant'],
            'fr': ['excellent', 'génial', 'bon', 'incroyable', 'adore', 'j\'adore',
                   'parfait', 'merveilleux', 'fantastique', 'meilleur',
                   'qualité', 'recommande', 'fiable', 'efficace', 'magnifique',
                   'impressionnant', 'super', 'formidable', 'ravie', 'satisfait']
        }

        negative_keywords = {
            'en': ['bad', 'poor', 'terrible', 'hate', 'worst', 'disappointing',
                   'problem', 'issue', 'expensive', 'difficult', 'slow',
                   'lack', 'insufficient', 'failure', 'error', 'wrong', 'broken'],
            'fr': ['mauvais', 'terrible', 'déteste', 'pire', 'décevant',
                   'problème', 'cher', 'difficile', 'lent', 'manque',
                   'insuffisant', 'échec', 'erreur', 'défaut', 'critique']
        }

        # IMPORTANT: Exclude patterns that indicate mixed or neutral sentiment
        exclude_patterns = {
            'en': ['but', 'however', 'although', 'despite', 'except'],
            'fr': ['mais', 'cependant', 'toutefois', 'malgré', 'sauf', 'bien que']
        }

        # Detect language
        lang = 'fr' if any(word in text.lower() for word in ['le', 'la', 'les', 'de']) else 'en'
        pos_words = positive_keywords[lang]
        neg_words = negative_keywords[lang]
        exclude_words = exclude_patterns[lang]

        for sent in manual_sentences:
            sent = sent.strip()

            # Skip if too short or already included
            if len(sent) < 30 or sent in positive_sentences or sent in negative_sentences:
                continue

            sent_lower = sent.lower()

            # Skip mixed sentiment sentences
            if any(word in sent_lower for word in exclude_words):
                continue

            # Count positive and negative words with WEIGHTS
            pos_count = sum(2 if word in sent_lower else 0 for word in pos_words)
            neg_count = sum(2 if word in sent_lower else 0 for word in neg_words)

            # Additional context checks
            # Check for negation patterns
            negation_patterns = ['pas', 'ne', 'non', 'aucun', 'not', 'no', 'never']
            has_negation = any(pattern in sent_lower for pattern in negation_patterns)

            # If negation is present, reverse the sentiment
            if has_negation:
                pos_count, neg_count = neg_count, pos_count

            # Add to appropriate list with STRONGER threshold
            if pos_count >= 2 and pos_count > neg_count * 2 and len(positive_sentences) < 8:
                positive_sentences.append(sent)
            elif neg_count >= 2 and neg_count > pos_count * 2 and len(negative_sentences) < 8:
                negative_sentences.append(sent)

    def _fallback_sentiment_analysis(self, text: str) -> Dict[str, Any]:
        """
        Fallback sentiment analysis using keyword matching

        Args:
            text: Text to analyze

        Returns:
            Dictionary with basic sentiment analysis including REAL counts
        """
        print("• Using fallback sentiment analysis")

        # Keyword lists for different languages
        positive_keywords = {
            'en': ['excellent', 'great', 'good', 'amazing', 'best', 'love', 'perfect',
                   'wonderful', 'fantastic', 'superior', 'outstanding', 'exceptional',
                   'brilliant', 'superb', 'magnificent', 'marvelous', 'delightful'],
            'fr': ['excellent', 'génial', 'bon', 'incroyable', 'meilleur', 'adore',
                   'parfait', 'merveilleux', 'fantastique', 'supérieur', 'exceptionnel',
                   'brillant', 'superbe', 'magnifique', 'formidable']
        }

        negative_keywords = {
            'en': ['bad', 'poor', 'worst', 'terrible', 'hate', 'awful', 'disappointing',
                   'failure', 'problem', 'issue', 'concern', 'negative', 'horrible',
                   'disgusting', 'unacceptable', 'mediocre', 'inferior'],
            'fr': ['mauvais', 'pire', 'terrible', 'déteste', 'horrible', 'décevant',
                   'échec', 'problème', 'préoccupation', 'négatif', 'dégoûtant',
                   'inacceptable', 'médiocre', 'inférieur']
        }

        # Detect language (simple approach)
        lang = 'fr' if any(word in text.lower() for word in ['le', 'la', 'les', 'de', 'du']) else 'en'

        pos_words = positive_keywords.get(lang, positive_keywords['en'])
        neg_words = negative_keywords.get(lang, negative_keywords['en'])

        text_lower = text.lower()
        positive_count = sum(1 for word in pos_words if word in text_lower)
        negative_count = sum(1 for word in neg_words if word in text_lower)

        # Calculate polarity
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
            pos_count = sum(1 for word in pos_words if word in sentence_lower)
            neg_count = sum(1 for word in neg_words if word in sentence_lower)

            if pos_count > neg_count:
                positive_sentences.append(sentence)
            elif neg_count > pos_count:
                negative_sentences.append(sentence)

        return {
            "polarity": polarity,
            "magnitude": abs(polarity),
            "subjectivity": 0.5,
            "sentiment": sentiment,
            "positive_aspects": positive_sentences[:5],  # Exemples limités
            "negative_aspects": negative_sentences[:5],  # Exemples limités
            "positive_count": len(positive_sentences),  # COMPTE RÉEL
            "negative_count": len(negative_sentences),  # COMPTE RÉEL
            "api_used": "fallback"
        }


def create_entity_visualization(entities: List[Dict[str, Any]]) -> go.Figure:
    """Create a visualization of extracted entities"""

    if not entities:
        return None

    # Prepare data
    names = [e['name'] for e in entities[:10]]
    saliences = [e['salience'] for e in entities[:10]]
    types = [e['type'] for e in entities[:10]]

    # Color mapping for entity types
    color_map = {
        'PERSON': '#FF6B6B',
        'ORGANIZATION': '#4ECDC4',
        'LOCATION': '#45B7D1',
        'EVENT': '#96CEB4',
        'WORK_OF_ART': '#DDA0DD',
        'CONSUMER_GOOD': '#F4A460',
        'OTHER': '#95A5A6'
    }

    colors = [color_map.get(t, '#95A5A6') for t in types]

    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=saliences,
            y=names,
            orientation='h',
            marker_color=colors,
            text=[f"{s:.3f}" for s in saliences],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Type: %{customdata}<br>Salience: %{x:.3f}<extra></extra>',
            customdata=types
        )
    ])

    fig.update_layout(
        title="Key Entities Mentioned",
        xaxis_title="Salience Score",
        yaxis_title="Entity",
        height=400,
        margin=dict(l=150),
        showlegend=False
    )

    return fig


def create_sentiment_magnitude_scatter(platform_sentiments: Dict[str, Dict]) -> go.Figure:
    """Create a scatter plot of sentiment vs magnitude by platform"""

    platforms = []
    sentiments = []
    magnitudes = []
    colors = []

    color_map = {
        'chatgpt': '#FF6B6B',
        'perplexity': '#4ECDC4',
        'gemini': '#45B7D1'
    }

    for platform, data in platform_sentiments.items():
        platforms.append(platform.title())
        sentiments.append(data.get('polarity', 0))
        magnitudes.append(data.get('magnitude', abs(data.get('polarity', 0))))
        colors.append(color_map.get(platform, '#95A5A6'))

    fig = go.Figure(data=[
        go.Scatter(
            x=sentiments,
            y=magnitudes,
            mode='markers+text',
            marker=dict(size=20, color=colors),
            text=platforms,
            textposition="top center",
            hovertemplate='<b>%{text}</b><br>Sentiment: %{x:.3f}<br>Magnitude: %{y:.3f}<extra></extra>'
        )
    ])

    # Add quadrant lines
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)

    # Add quadrant labels
    fig.add_annotation(x=0.5, y=0.9, text="Positive & Strong", showarrow=False, opacity=0.7)
    fig.add_annotation(x=-0.5, y=0.9, text="Negative & Strong", showarrow=False, opacity=0.7)
    fig.add_annotation(x=0.5, y=0.1, text="Positive & Weak", showarrow=False, opacity=0.7)
    fig.add_annotation(x=-0.5, y=0.1, text="Negative & Weak", showarrow=False, opacity=0.7)

    fig.update_layout(
        title="Sentiment Analysis by Platform",
        xaxis_title="Sentiment (Negative ← → Positive)",
        yaxis_title="Magnitude (Strength of Emotion)",
        xaxis=dict(range=[-1, 1]),
        yaxis=dict(range=[0, 1]),
        height=500
    )

    return fig