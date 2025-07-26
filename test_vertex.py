"""
Test simple pour vérifier la connexion à Google Cloud Natural Language API
À exécuter directement pour diagnostiquer le problème
"""

import os
import json

# Test 1: Vérifier l'installation
print("=== TEST 1: Vérification des imports ===")
try:
    from google.cloud import language_v1
    from google.oauth2 import service_account

    print("✅ Google Cloud libraries importées avec succès")
except ImportError as e:
    print("❌ Erreur d'import:", e)
    print("Installez avec: pip install google-cloud-language")
    exit(1)

# Test 2: Vérifier le fichier de credentials
print("\n=== TEST 2: Vérification du fichier credentials ===")
credentials_path = "credentials/vertexai-465711-e8ceb761e644.json"

# Afficher le chemin complet
full_path = os.path.abspath(credentials_path)
print(f"Chemin complet: {full_path}")
print(f"Fichier existe: {os.path.exists(credentials_path)}")

if not os.path.exists(credentials_path):
    print("❌ Fichier non trouvé!")
    print("Créez le dossier 'credentials' et placez-y le fichier JSON")
    exit(1)

# Test 3: Vérifier le contenu du fichier
print("\n=== TEST 3: Vérification du contenu JSON ===")
try:
    with open(credentials_path, 'r') as f:
        creds_data = json.load(f)
        print(f"✅ Project ID: {creds_data.get('project_id')}")
        print(f"✅ Service Account: {creds_data.get('client_email')}")
        print(f"✅ Private Key ID: {creds_data.get('private_key_id')[:20]}...")
except Exception as e:
    print(f"❌ Erreur lecture JSON: {e}")
    exit(1)

# Test 4: Créer le client
print("\n=== TEST 4: Création du client ===")
try:
    # Méthode 1: Avec credentials explicites
    credentials = service_account.Credentials.from_service_account_file(
        credentials_path,
        scopes=['https://www.googleapis.com/auth/cloud-language']
    )
    client = language_v1.LanguageServiceClient(credentials=credentials)
    print("✅ Client créé avec succès")
except Exception as e:
    print(f"❌ Erreur création client: {e}")
    print(f"Type d'erreur: {type(e).__name__}")
    exit(1)

# Test 5: Tester l'API
print("\n=== TEST 5: Test de l'API ===")
try:
    # Texte de test simple
    test_text = "Google Cloud Natural Language API is amazing!"

    document = language_v1.Document(
        content=test_text,
        type_=language_v1.Document.Type.PLAIN_TEXT,
        language="en"
    )

    print(f"📝 Analyse du texte: '{test_text}'")

    # Appel API
    response = client.analyze_sentiment(
        request={'document': document}
    )

    sentiment = response.document_sentiment
    print(f"✅ Analyse réussie!")
    print(f"   Score: {sentiment.score}")
    print(f"   Magnitude: {sentiment.magnitude}")

except Exception as e:
    print(f"❌ Erreur API: {e}")
    print(f"Type: {type(e).__name__}")

    # Diagnostic détaillé
    error_str = str(e)
    if "403" in error_str:
        print("\n💡 Solutions possibles:")
        print("1. Vérifiez que l'API est activée dans le bon projet")
        print("2. Vérifiez que le compte de service a les bonnes permissions")
        print("3. Essayez d'ajouter le rôle 'Cloud Natural Language User'")
    elif "401" in error_str:
        print("\n💡 Problème d'authentification - vérifiez vos credentials")
    elif "quota" in error_str.lower():
        print("\n💡 Problème de quota - vérifiez dans la console Google Cloud")

# Test 6: Alternative avec variable d'environnement
print("\n=== TEST 6: Test avec variable d'environnement ===")
try:
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
    client2 = language_v1.LanguageServiceClient()
    print("✅ Client créé via variable d'environnement")
except Exception as e:
    print(f"❌ Erreur avec env var: {e}")

print("\n=== DIAGNOSTIC TERMINÉ ===")


# FONCTION CORRIGÉE pour vertex_sentiment_analyzer.py
def create_vertex_client_fixed(credentials_path):
    """Version corrigée pour créer le client"""
    try:
        # S'assurer que le chemin est absolu
        if not os.path.isabs(credentials_path):
            credentials_path = os.path.abspath(credentials_path)

        if not os.path.exists(credentials_path):
            raise FileNotFoundError(f"Credentials not found: {credentials_path}")

        # Option 1: Credentials explicites (recommandé)
        from google.oauth2 import service_account
        from google.cloud import language_v1

        credentials = service_account.Credentials.from_service_account_file(
            credentials_path,
            scopes=['https://www.googleapis.com/auth/cloud-language']
        )

        client = language_v1.LanguageServiceClient(credentials=credentials)

        # Test rapide
        test_doc = language_v1.Document(
            content="Test",
            type_=language_v1.Document.Type.PLAIN_TEXT,
        )
        client.analyze_sentiment(request={'document': test_doc})

        return client, None

    except Exception as e:
        # Option 2: Variable d'environnement (fallback)
        try:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
            client = language_v1.LanguageServiceClient()
            return client, None
        except Exception as e2:
            return None, f"Failed both methods: {str(e)}, {str(e2)}"


# AMÉLIORATION pour brand_llm_analysis.py
def initialize_vertex_analyzer_improved(credentials_path):
    """Initialisation améliorée avec meilleur logging"""
    import streamlit as st

    # Log du chemin
    st.info(f"🔍 Tentative de connexion à Google Cloud NL API...")
    st.info(f"📁 Chemin credentials: {os.path.abspath(credentials_path)}")

    # Vérifier que le fichier existe
    if not os.path.exists(credentials_path):
        st.error(f"❌ Fichier credentials non trouvé: {credentials_path}")
        st.info("Créez un dossier 'credentials' à la racine du projet et placez-y votre fichier JSON")
        return None

    # Importer et initialiser
    try:
        from utils.vertex_sentiment_analyzer import VertexSentimentAnalyzer

        analyzer = VertexSentimentAnalyzer(credentials_path)

        if analyzer.available:
            st.success("✅ Google Cloud Natural Language API connectée!")
            # Faire un test rapide
            test_result = analyzer.analyze_sentiment("Test connection")
            if test_result.get('api_used') == 'google_cloud_nl':
                st.success("✅ Test d'analyse réussi avec Google Cloud!")
            else:
                st.warning(f"⚠️ Fallback utilisé: {test_result.get('api_used')}")
        else:
            st.error(f"❌ Connexion échouée: {analyzer.error_message}")
            st.info("L'analyse utilisera TextBlob comme méthode de secours")

        return analyzer

    except ImportError:
        st.error("❌ Module vertex_sentiment_analyzer non trouvé")
        return None
    except Exception as e:
        st.error(f"❌ Erreur: {str(e)}")
        return None