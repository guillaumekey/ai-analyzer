"""
Translation module for AI Visibility Audit Tool
"""

TRANSLATIONS = {
    "en": {
        # App info
        "app_title": "AI Visibility Audit Tool",
        "app_description": "Analyze your brand's visibility across ChatGPT, Perplexity, and Gemini AI platforms",
        "footer": "Built with ❤️ using Streamlit",

        # Sidebar
        "configuration": "Configuration",
        "api_keys": "API Keys",
        "api_key": "API Key",
        "model": "Model",
        "enter_api_key": "Enter your {platform} API key",
        "select_model": "Select the {platform} model to use",
        "active_platforms": "Active Platforms",
        "no_api_key": "no API key",
        "no_platforms": "No platforms configured",
        "platforms_ready": "{count} platform(s) ready",
        "display_options": "Display Options",
        "show_individual": "Show individual platform results",
        "display_detailed_help": "Display detailed results for each prompt on each platform",
        "language": "Language",

        # Main form
        "brand_information": "Brand Information",
        "brand_name": "Brand Name",
        "brand_name_placeholder": "e.g., Nike",
        "brand_url": "Brand URL",
        "brand_url_placeholder": "e.g., https://nike.com",
        "competitors": "Competitors (one per line) - Optional",
        "competitors_placeholder": "Adidas\nPuma\nReebok",
        "competitors_help": "Enter competitor brand names to track their mentions. Leave empty for auto-detection only.",
        "test_prompts": "Test Prompts",
        "enter_prompts": "Enter prompts (one per line)",
        "run_analysis": "Run Visibility Analysis",

        # Validation errors
        "error_no_brand": "Please enter a brand name",
        "error_no_prompts": "Please enter at least one prompt",
        "error_no_api_key": "Please enter at least one API key",

        # Progress messages
        "testing_platforms": "Testing on {count} platform(s): {platforms}",
        "analyzing_visibility": "Analyzing brand visibility...",
        "detecting_competitors": "Detecting competitors with AI...",
        "detected_competitors": "AI detected competitors: **{competitors}**",
        "analyzing_competitors": "Analyzing competitor mentions...",
        "analyzing_count": "Analyzing {count} competitors: {competitors}",
        "analysis_complete": "Analysis Complete!",

        # Results - Summary
        "visibility_summary": "Visibility Summary",
        "unique_mentions": "Unique Mentions",
        "total_mentions": "Total Mentions",
        "visibility_rate": "Visibility Rate",
        "prompts_tested": "Prompts Tested",
        "platforms": "Platforms",
        "platform_breakdown": "Platform Breakdown",
        "avg_per_prompt": "Avg per Prompt",

        # Results - Competitive Analysis
        "competitive_analysis": "Competitive Analysis",
        "configure_competitors": "Configure Competitors",
        "select_competitors": "Select which competitors to include in the analysis:",
        "selected_count": "Selected {selected} out of {total} competitors",
        "add_competitor": "Add another competitor:",
        "add": "Add",
        "added_competitor": "Added {competitor} to analysis",
        "no_competitors": "No competitors detected or specified. Add competitors to see comparison analysis.",
        "select_one_competitor": "Please select at least one competitor to analyze.",

        # Competitive metrics
        "your_brand_presence": "Your Brand Presence",
        "avg_competitor_presence": "Avg Competitor Presence",
        "your_ranking": "Your Ranking",
        "total_brands": "Total Brands Analyzed",
        "brand_ranking": "Brand Ranking",
        "gap_to_leader": "Gap to Leader",
        "lead_over_2": "Lead Over #2",
        "strongest_platform": "Strongest Platform",
        "mention_density": "Mention Density",
        "behind_leader": "{n} behind leader",
        "prompts": "prompts",

        # Competitive insights
        "key_insights": "Key Insights",
        "recommendations": "Recommendations",
        "improve_visibility": """**To improve your visibility:**
        - Your brand appears in {unique} prompts ({rate:.1f}% visibility rate)
        - The leader appears in {leader_unique} prompts ({leader_rate:.1f}% visibility rate)
        - Focus on increasing presence in prompts where you're not mentioned
        - Analyze prompt types where competitors appear but you don't""",
        "maintain_leadership": """**Maintain your leadership position:**
        - Your brand leads with presence in {unique} prompts
        - You have a {rate:.1f}% visibility rate across all prompts
        - Continue optimizing for consistent presence across all prompt types
        - Monitor competitors' strategies to maintain your lead""",

        # Tables and charts
        "detailed_comparison": "Detailed Mentions Comparison",
        "format_note": "Format: Unique Mentions / Total Mentions",
        "presence_comparison": "Presence Comparison",
        "visibility_rates": "Visibility Rates",
        "platform_dominance": "Platform Dominance",
        "presence_heatmap": "Presence Heatmap",
        "total_mentions_chart": "Total Mentions",
        "all_occurrences": "This chart shows all occurrences (multiple mentions per prompt)",

        # Chart titles
        "brand_presence_title": "Brand Presence Comparison Across AI Platforms (Unique Mentions per Prompt)",
        "total_mentions_title": "Total Brand Mentions Across AI Platforms (All Occurrences)",
        "visibility_rate_title": "Visibility Rate Comparison (% of Prompts with at Least One Mention)",
        "platform_dominance_title": "Platform Dominance Comparison (Prompts with Mentions)",
        "heatmap_title": "Brand Presence Heatmap - {platform}",
        "dominance_subtitle": "This chart shows how many prompts each brand appears in, by platform",

        # Heatmap
        "analyzing_prompts": "Analyzing {count} prompts across {brands} brands",
        "heatmap_view": "Heatmap View",
        "summary_statistics": "Summary Statistics",
        "prompt_details": "Prompt Details",
        "heatmap_tip": "Tip: With {count} prompts, consider using the Summary Statistics tab for a clearer overview.",
        "presence_summary": "Presence Summary by Platform",
        "prompts_with_presence": "Prompts with Presence",
        "presence_rate": "Presence Rate",
        "search_prompts": "Search prompts:",
        "search_placeholder": "Enter keywords to filter prompts...",
        "select_platform": "Select Platform:",
        "showing_prompts": "Showing {filtered} of {total} prompts",
        "prompt": "Prompt",
        "showing_first_50": "Showing first 50 prompts only. Use search to find specific prompts.",

        # Sources analysis
        "sources_analysis": "Perplexity Sources Analysis",
        "domain": "Domain",
        "count": "Count",
        "source_url": "Source URL",
        "different_pages": "{count} different pages",
        "view_all_urls": "View all URLs for domains with multiple pages",
        "total_sources": "Total Sources",
        "unique_domains": "Unique Domains",
        "unique_urls": "Unique URLs",
        "export_sources": "Export Full Source List",
        "download_csv": "Download Sources CSV",
        "no_sources": "No sources found in Perplexity responses",

        # Export
        "export_results": "Export Results",
        "download_report": "Download Full Report (JSON)",

        # Debug
        "debug_competitor": "Debug - Competitor Data",
        "has_competitor_mentions": "Has competitor_mentions:",
        "competitor_mentions": "Competitor mentions:",

        # Brand LLM Analysis
        "brand_llm_analysis": "Brand LLM Analysis",
        "run_brand_analysis": "Run Brand LLM Analysis",
        "brand_analysis_help": "Deep dive into what LLMs know about your brand and its reputation",
        "analyzing_brand_knowledge": "Analyzing brand knowledge...",
        "analyzing_brand_reputation": "Analyzing brand reputation...",
        "processing_sentiment": "Processing sentiment and themes...",
        "overall_sentiment": "Overall Sentiment",
        "subjectivity_score": "Subjectivity Score",
        "key_themes": "Key Themes",
        "aspects_covered": "Aspects Covered",
        "sentiment_analysis_tab": "Sentiment Analysis",
        "key_themes_tab": "Key Themes",
        "brand_aspects_tab": "Brand Aspects",
        "knowledge_summary_tab": "Knowledge Summary",
        "reputation_summary_tab": "Reputation Summary",
        "overall_brand_sentiment": "Overall Brand Sentiment",
        "sentiment_by_platform": "Sentiment by Platform",
        "positive_aspects": "Positive Aspects",
        "negative_aspects": "Negative Aspects",
        "brand_theme_cloud": "Brand Theme Cloud",
        "top_themes": "Top 15 Key Themes",
        "brand_aspect_coverage": "Brand Aspect Coverage",
        "detailed_brand_aspects": "Detailed Brand Aspects",
        "what_llms_know": "What LLMs Know About",
        "brand_reputation_analysis": "Brand Reputation Analysis",
        "export_brand_analysis": "Export Brand Analysis Report",
        "download_brand_analysis": "Download Brand Analysis JSON",
        "theme": "Theme",
        "frequency": "Frequency",
        "most_frequent_themes": "Most Frequent Themes",
        "sentiment_score": "Sentiment Score",
        "platform": "Platform",
        "query": "Query",
    },

    "fr": {
        # App info
        "app_title": "Outil d'Audit de Visibilité IA",
        "app_description": "Analysez la visibilité de votre marque sur ChatGPT, Perplexity et Gemini",
        "footer": "Créé avec ❤️ avec Streamlit",

        # Sidebar
        "configuration": "Configuration",
        "api_keys": "Clés API",
        "api_key": "Clé API",
        "model": "Modèle",
        "enter_api_key": "Entrez votre clé API {platform}",
        "select_model": "Sélectionnez le modèle {platform} à utiliser",
        "active_platforms": "Plateformes Actives",
        "no_api_key": "pas de clé API",
        "no_platforms": "Aucune plateforme configurée",
        "platforms_ready": "{count} plateforme(s) prête(s)",
        "display_options": "Options d'Affichage",
        "show_individual": "Afficher les résultats individuels par plateforme",
        "display_detailed_help": "Afficher les résultats détaillés pour chaque prompt sur chaque plateforme",
        "language": "Langue",

        # Main form
        "brand_information": "Informations de la Marque",
        "brand_name": "Nom de la Marque",
        "brand_name_placeholder": "ex: Nike",
        "brand_url": "URL de la Marque",
        "brand_url_placeholder": "ex: https://nike.com",
        "competitors": "Concurrents (un par ligne) - Optionnel",
        "competitors_placeholder": "Adidas\nPuma\nReebok",
        "competitors_help": "Entrez les noms des marques concurrentes. Laissez vide pour la détection automatique uniquement.",
        "test_prompts": "Prompts de Test",
        "enter_prompts": "Entrez les prompts (un par ligne)",
        "run_analysis": "Lancer l'Analyse de Visibilité",

        # Validation errors
        "error_no_brand": "Veuillez entrer un nom de marque",
        "error_no_prompts": "Veuillez entrer au moins un prompt",
        "error_no_api_key": "Veuillez entrer au moins une clé API",

        # Progress messages
        "testing_platforms": "Test sur {count} plateforme(s) : {platforms}",
        "analyzing_visibility": "Analyse de la visibilité de la marque...",
        "detecting_competitors": "Détection des concurrents avec l'IA...",
        "detected_competitors": "L'IA a détecté les concurrents : **{competitors}**",
        "analyzing_competitors": "Analyse des mentions des concurrents...",
        "analyzing_count": "Analyse de {count} concurrents : {competitors}",
        "analysis_complete": "Analyse Terminée !",

        # Results - Summary
        "visibility_summary": "Résumé de Visibilité",
        "unique_mentions": "Mentions Uniques",
        "total_mentions": "Mentions Totales",
        "visibility_rate": "Taux de Visibilité",
        "prompts_tested": "Prompts Testés",
        "platforms": "Plateformes",
        "platform_breakdown": "Répartition par Plateforme",
        "avg_per_prompt": "Moy. par Prompt",

        # Results - Competitive Analysis
        "competitive_analysis": "Analyse Concurrentielle",
        "configure_competitors": "Configurer les Concurrents",
        "select_competitors": "Sélectionnez les concurrents à inclure dans l'analyse :",
        "selected_count": "{selected} sélectionnés sur {total} concurrents",
        "add_competitor": "Ajouter un autre concurrent :",
        "add": "Ajouter",
        "added_competitor": "{competitor} ajouté à l'analyse",
        "no_competitors": "Aucun concurrent détecté ou spécifié. Ajoutez des concurrents pour voir l'analyse comparative.",
        "select_one_competitor": "Veuillez sélectionner au moins un concurrent à analyser.",

        # Competitive metrics
        "your_brand_presence": "Présence de Votre Marque",
        "avg_competitor_presence": "Présence Moy. Concurrents",
        "your_ranking": "Votre Classement",
        "total_brands": "Total Marques Analysées",
        "brand_ranking": "Classement Marque",
        "gap_to_leader": "Écart au Leader",
        "lead_over_2": "Avance sur le #2",
        "strongest_platform": "Plateforme la Plus Forte",
        "mention_density": "Densité de Mentions",
        "behind_leader": "{n} derrière le leader",
        "prompts": "prompts",

        # Competitive insights
        "key_insights": "Points Clés",
        "recommendations": "Recommandations",
        "improve_visibility": """**Pour améliorer votre visibilité :**
        - Votre marque apparaît dans {unique} prompts (taux de visibilité de {rate:.1f}%)
        - Le leader apparaît dans {leader_unique} prompts (taux de visibilité de {leader_rate:.1f}%)
        - Concentrez-vous sur l'augmentation de votre présence dans les prompts où vous n'êtes pas mentionné
        - Analysez les types de prompts où les concurrents apparaissent mais pas vous""",
        "maintain_leadership": """**Maintenez votre position de leader :**
        - Votre marque est en tête avec une présence dans {unique} prompts
        - Vous avez un taux de visibilité de {rate:.1f}% sur tous les prompts
        - Continuez à optimiser pour une présence constante sur tous les types de prompts
        - Surveillez les stratégies des concurrents pour maintenir votre avance""",

        # Tables and charts
        "detailed_comparison": "Comparaison Détaillée des Mentions",
        "format_note": "Format : Mentions Uniques / Mentions Totales",
        "presence_comparison": "Comparaison de Présence",
        "visibility_rates": "Taux de Visibilité",
        "platform_dominance": "Dominance par Plateforme",
        "presence_heatmap": "Carte de Présence",
        "total_mentions_chart": "Mentions Totales",
        "all_occurrences": "Ce graphique montre toutes les occurrences (mentions multiples par prompt)",

        # Chart titles
        "brand_presence_title": "Comparaison de Présence des Marques sur les Plateformes IA (Mentions Uniques par Prompt)",
        "total_mentions_title": "Mentions Totales des Marques sur les Plateformes IA (Toutes Occurrences)",
        "visibility_rate_title": "Comparaison des Taux de Visibilité (% de Prompts avec au Moins une Mention)",
        "platform_dominance_title": "Comparaison de Dominance par Plateforme (Prompts avec Mentions)",
        "heatmap_title": "Carte de Présence des Marques - {platform}",
        "dominance_subtitle": "Ce graphique montre dans combien de prompts chaque marque apparaît, par plateforme",

        # Heatmap
        "analyzing_prompts": "Analyse de {count} prompts sur {brands} marques",
        "heatmap_view": "Vue Carte de Chaleur",
        "summary_statistics": "Statistiques Résumées",
        "prompt_details": "Détails des Prompts",
        "heatmap_tip": "Conseil : Avec {count} prompts, utilisez l'onglet Statistiques Résumées pour une vue plus claire.",
        "presence_summary": "Résumé de Présence par Plateforme",
        "prompts_with_presence": "Prompts avec Présence",
        "presence_rate": "Taux de Présence",
        "search_prompts": "Rechercher des prompts :",
        "search_placeholder": "Entrez des mots-clés pour filtrer...",
        "select_platform": "Sélectionner la Plateforme :",
        "showing_prompts": "Affichage de {filtered} prompts sur {total}",
        "prompt": "Prompt",
        "showing_first_50": "Affichage des 50 premiers prompts. Utilisez la recherche pour trouver des prompts spécifiques.",

        # Sources analysis
        "sources_analysis": "Analyse des Sources Perplexity",
        "domain": "Domaine",
        "count": "Nombre",
        "source_url": "URL Source",
        "different_pages": "{count} pages différentes",
        "view_all_urls": "Voir toutes les URLs pour les domaines avec plusieurs pages",
        "total_sources": "Total Sources",
        "unique_domains": "Domaines Uniques",
        "unique_urls": "URLs Uniques",
        "export_sources": "Exporter la Liste Complète",
        "download_csv": "Télécharger CSV des Sources",
        "no_sources": "Aucune source trouvée dans les réponses Perplexity",

        # Export
        "export_results": "Exporter les Résultats",
        "download_report": "Télécharger le Rapport Complet (JSON)",

        # Debug
        "debug_competitor": "Debug - Données Concurrents",
        "has_competitor_mentions": "A competitor_mentions :",
        "competitor_mentions": "Mentions concurrents :",

        # Brand LLM Analysis
        "brand_llm_analysis": "Analyse LLM de la Marque",
        "run_brand_analysis": "Lancer l'Analyse LLM de la Marque",
        "brand_analysis_help": "Analyse approfondie de ce que les LLMs savent sur votre marque et sa réputation",
        "analyzing_brand_knowledge": "Analyse des connaissances sur la marque...",
        "analyzing_brand_reputation": "Analyse de la réputation de la marque...",
        "processing_sentiment": "Traitement du sentiment et des thèmes...",
        "overall_sentiment": "Sentiment Général",
        "subjectivity_score": "Score de Subjectivité",
        "key_themes": "Thèmes Clés",
        "aspects_covered": "Aspects Couverts",
        "sentiment_analysis_tab": "Analyse de Sentiment",
        "key_themes_tab": "Thèmes Clés",
        "brand_aspects_tab": "Aspects de la Marque",
        "knowledge_summary_tab": "Résumé des Connaissances",
        "reputation_summary_tab": "Résumé de Réputation",
        "overall_brand_sentiment": "Sentiment Général de la Marque",
        "sentiment_by_platform": "Sentiment par Plateforme",
        "positive_aspects": "Aspects Positifs",
        "negative_aspects": "Aspects Négatifs",
        "brand_theme_cloud": "Nuage de Thèmes de la Marque",
        "top_themes": "Top 15 Thèmes Clés",
        "brand_aspect_coverage": "Couverture des Aspects de la Marque",
        "detailed_brand_aspects": "Aspects Détaillés de la Marque",
        "what_llms_know": "Ce que les LLMs Savent sur",
        "brand_reputation_analysis": "Analyse de Réputation de la Marque",
        "export_brand_analysis": "Exporter le Rapport d'Analyse",
        "download_brand_analysis": "Télécharger l'Analyse JSON",
        "theme": "Thème",
        "frequency": "Fréquence",
        "most_frequent_themes": "Thèmes les Plus Fréquents",
        "sentiment_score": "Score de Sentiment",
        "platform": "Plateforme",
        "query": "Requête",

        # Export
        "export_results": "Export Results",
        "download_report": "Download Full Report (JSON)",
        "download_pdf_report": "Download PDF Report",
        "generating_pdf": "Generating PDF report...",
    }
}


def get_text(key: str, lang: str = "en", **kwargs) -> str:
    """
    Get translated text for a given key.

    Args:
        key: The translation key
        lang: Language code ('en' or 'fr')
        **kwargs: Format arguments for string formatting

    Returns:
        Translated string
    """
    try:
        text = TRANSLATIONS.get(lang, TRANSLATIONS["en"]).get(key, key)
        if kwargs:
            return text.format(**kwargs)
        return text
    except:
        return key