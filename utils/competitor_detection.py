"""
AI-powered competitor detection utilities - Enhanced version
"""
from typing import List, Dict, Set
import json
from collections import Counter
import re


def detect_competitors_with_ai(client, responses_text: str, brand_name: str, max_competitors: int = 50) -> List[str]:
    """Use AI to detect competitors mentioned in the responses - ENHANCED VERSION"""

    # More specific prompt to catch ALL brand names
    prompt = f"""Analyze the following AI responses about "{brand_name}" and extract EVERY SINGLE brand or company name mentioned.

RESPONSES TO ANALYZE:
{responses_text}

INSTRUCTIONS:
1. Extract EVERY brand name or company name mentioned in the text
2. Include brands mentioned in ANY context:
   - In lists (numbered or bulleted)
   - In comparisons
   - In recommendations
   - In examples
   - As alternatives
   - In parentheses
   - With descriptions
3. Include ALL types of companies and brands from ANY industry:
   - Technology companies (e.g., Apple, Microsoft, Samsung)
   - Fashion brands (e.g., Nike, Adidas, Zara)
   - Food & beverage brands (e.g., Coca-Cola, McDonald's, Nestlé)
   - Automotive brands (e.g., Toyota, Tesla, BMW)
   - Service companies (e.g., Uber, Airbnb, Netflix)
   - B2B companies (e.g., Salesforce, SAP, Oracle)
   - Any other brand or company name regardless of industry
4. Extract the brand name even if it appears with additional text like:
   - Special characters (e.g., "AT&T", "Procter & Gamble")
   - Numbers (e.g., "3M", "7-Eleven")
   - Unusual capitalization (e.g., "eBay", "iPhone", "LinkedIn")
   - Multiple words (e.g., "General Electric", "Johnson & Johnson")
5. Do NOT include:
   - The brand "{brand_name}" itself
   - Generic terms like "other brands", "competitors", "leading companies"
   - Descriptive words that aren't brand names
   - Common nouns or general product categories
6. Return ONLY a JSON array of ALL brand names found
7. Be EXHAUSTIVE - if you see a proper noun that could be a brand or company, include it

Example output format: ["Apple", "Samsung", "Nike", "Coca-Cola", "Tesla", ...]

Return ONLY the JSON array, no other text."""

    try:
        # Call the AI to analyze competitors
        if hasattr(client, 'client'):  # OpenAI client
            response = client.client.chat.completions.create(
                model=client.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # Low temperature for consistent extraction
                max_tokens=2000  # Increased for more competitors
            )
            result = response.choices[0].message.content.strip()
        else:
            # Fallback for other clients
            result = client.call_api(prompt)

        # Parse the JSON response
        try:
            # Clean up the response to ensure it's valid JSON
            result = result.strip()
            if result.startswith('```json'):
                result = result[7:]
            if result.startswith('```'):
                result = result[3:]
            if result.endswith('```'):
                result = result[:-3]

            competitors = json.loads(result.strip())

            # Ensure it's a list and filter out invalid entries
            if isinstance(competitors, list):
                # Clean and validate each competitor name
                valid_competitors = []
                seen = set()  # Avoid duplicates

                for comp in competitors:
                    if isinstance(comp, str) and comp.strip():
                        cleaned = comp.strip()
                        # Skip if it's the brand itself (case insensitive)
                        if cleaned.lower() != brand_name.lower() and cleaned.lower() not in seen:
                            valid_competitors.append(cleaned)
                            seen.add(cleaned.lower())

                return valid_competitors[:max_competitors]
            else:
                return []

        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract brand names manually
            print(f"JSON parsing failed, using fallback extraction")
            return extract_brands_from_text(responses_text, brand_name)

    except Exception as e:
        print(f"Error detecting competitors: {e}")
        # Fallback to pattern matching
        return extract_brands_from_text(responses_text, brand_name)


def extract_brands_from_text(text: str, brand_name: str) -> List[str]:
    """Extract brand names from text using enhanced pattern matching"""

    brands = set()

    # Enhanced patterns for brand detection
    patterns = [
        # Numbered lists (1. Brand, 2. Brand)
        r'\d+\.\s*([A-Z][A-Za-z\s\+\&\'\.]+?)(?:\s*[:–—-]|\s*$|\n)',

        # Bullet points
        r'[•·]\s*([A-Z][A-Za-z\s\+\&\'\.]+?)(?:\s*[:–—-]|\s*$|\n)',

        # Bold text (common in formatted responses)
        r'\*\*([A-Z][A-Za-z\s\+\&\'\.]+?)\*\*',

        # Brands with descriptions
        r'([A-Z][A-Za-z\s\+\&\'\.]+?)\s*(?:–|—|-)\s*[A-Z][a-z]',

        # Brands in quotes
        r'"([A-Z][A-Za-z\s\+\&\'\.]+?)"',
        r"'([A-Z][A-Za-z\s\+\&\'\.]+?)'",

        # Common patterns like "Brand offers", "Brand is known"
        r'([A-Z][A-Za-z\s\+\&\'\.]+?)\s+(?:offers|is known|provides|features|has|makes)',

        # Capture brands before parentheses
        r'([A-Z][A-Za-z\s\+\&\'\.]+?)\s*\(',

        # Lists with commas
        r'(?:include|including|such as|like)\s+([A-Z][A-Za-z\s\+\&\'\.]+?)(?:\s*,|\s*and|\s*or|\s*\.)',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.MULTILINE)
        for match in matches:
            # Clean the match
            cleaned = match.strip()
            cleaned = re.sub(r'\s+', ' ', cleaned)  # Normalize spaces

            # Filter out common non-brand words
            non_brands = {
                'The', 'This', 'These', 'Those', 'They', 'For', 'From', 'With',
                'Best', 'Top', 'Great', 'Good', 'New', 'Brand', 'Brands',
                'Company', 'Companies', 'Store', 'Stores', 'Shop', 'Shops',
                'Known', 'Famous', 'Popular', 'Leading', 'Major', 'Main',
                'First', 'Second', 'Third', 'Last', 'Next', 'Other',
                'High', 'Low', 'Medium', 'Small', 'Large', 'Big',
                'Local', 'Global', 'National', 'International', 'Regional',
                'Based', 'Founded', 'Established', 'Created', 'Developed'
            }

            # Check if it's likely a brand
            if (cleaned and
                    cleaned not in non_brands and
                    cleaned.lower() != brand_name.lower() and
                    len(cleaned) > 2 and
                    not cleaned.isupper() and  # Avoid acronyms like "USA"
                    any(c.isupper() for c in cleaned)):  # Has at least one capital letter

                brands.add(cleaned)

    # Additional specific extraction for complex brand names
    special_patterns = [
        # Brands with special characters (AT&T, Procter & Gamble, Alice + Olivia)
        r'([A-Z][A-Za-z0-9]*\s*[+&]\s*[A-Z][A-Za-z0-9]+)',
        # Brands with numbers (3M, 7-Eleven)
        r'([0-9]+[A-Z][A-Za-z]*|[A-Z][A-Za-z]*[0-9]+)',
        # CamelCase brands (eBay, LinkedIn, LoveShackFancy)
        r'([a-z]+[A-Z][a-z]+(?:[A-Z][a-z]+)*)',
        r'([A-Z][a-z]+[A-Z][a-z]+(?:[A-Z][a-z]+)*)',
        # Brands with dots (St. Agni, Dr. Martens)
        r'([A-Z][a-z]*\.\s*[A-Z][a-z]+)',
        # Multi-word brands (General Electric, Procter & Gamble)
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)(?=\s+(?:is|are|was|were|has|have|offers|provides))',
        # Hyphenated brands (Rolls-Royce, Mercedes-Benz)
        r'([A-Z][a-z]+(?:-[A-Z][a-z]+)+)',
        # Brands ending with common suffixes
        r'([A-Z][A-Za-z]+(?:Corp|Inc|Ltd|LLC|GmbH|SA|SpA|AG|Co)\.?)\b',
    ]

    for pattern in special_patterns:
        special_matches = re.findall(pattern, text)
        for match in special_matches:
            if match.lower() != brand_name.lower():
                brands.add(match)

    return list(brands)


def aggregate_responses_for_detection(results: Dict) -> str:
    """Aggregate all responses into a single text for competitor detection"""
    all_responses = []

    for platform_name, platform_data in results.items():
        for response_data in platform_data.get('responses', []):
            response_text = response_data.get('response', '')
            if response_text and not response_text.startswith('Error'):
                # Include full response for better detection
                all_responses.append(f"[{platform_name}] {response_text}")

    return "\n\n".join(all_responses)


def detect_competitors_from_results(ai_client, results: Dict, brand_name: str,
                                    existing_competitors: List[str] = None) -> Dict[str, List[str]]:
    """Detect competitors from all AI responses using AI analysis"""

    # Aggregate all responses
    responses_text = aggregate_responses_for_detection(results)

    if not responses_text:
        return {'detected': [], 'new_suggestions': []}

    # Use AI to detect competitors
    detected_competitors = detect_competitors_with_ai(ai_client, responses_text, brand_name)

    # If AI detection fails or finds too few, try pattern matching as fallback
    if len(detected_competitors) < 3:
        fallback_competitors = extract_brands_from_text(responses_text, brand_name)
        # Merge with AI results, avoiding duplicates
        for comp in fallback_competitors:
            if comp not in detected_competitors:
                detected_competitors.append(comp)

    # Filter out existing competitors if provided
    if existing_competitors:
        existing_lower = [c.lower() for c in existing_competitors]
        new_suggestions = [c for c in detected_competitors if c.lower() not in existing_lower]
    else:
        new_suggestions = detected_competitors

    return {
        'detected': detected_competitors,
        'new_suggestions': new_suggestions
    }