"""
AI-powered competitor detection utilities - Enhanced version with better detection
"""
from typing import List, Dict, Set
import json
from collections import Counter
import re


def is_same_brand(candidate: str, brand_name: str) -> bool:
    """Check if candidate is the same as the brand being analyzed"""
    # Normalize for comparison
    candidate_normalized = candidate.lower().strip()
    brand_normalized = brand_name.lower().strip()

    # Exact match
    if candidate_normalized == brand_normalized:
        return True

    # Check without common suffixes
    suffixes = [' inc', ' inc.', ' corp', ' corp.', ' ltd', ' ltd.', ' llc', ' co', ' co.', ' company']
    for suffix in suffixes:
        if candidate_normalized.endswith(suffix):
            candidate_base = candidate_normalized[:-len(suffix)].strip()
            if candidate_base == brand_normalized:
                return True
        if brand_normalized.endswith(suffix):
            brand_base = brand_normalized[:-len(suffix)].strip()
            if candidate_normalized == brand_base:
                return True

    return False


def contains_brand_name(candidate: str, brand_name: str) -> bool:
    """Check if candidate contains the brand name as a primary identifier"""
    candidate_lower = candidate.lower()
    brand_lower = brand_name.lower()

    # Check if starts with brand name followed by space or common separators
    separators = [' ', '-', '_', '.']
    for sep in separators:
        if candidate_lower.startswith(brand_lower + sep):
            return True

    # Check exact match
    if candidate_lower == brand_lower:
        return True

    # Check if it's a product variant (e.g., "PLAUD NotePin" when brand is "Plaud")
    words = candidate.split()
    if words and words[0].lower() == brand_lower:
        return True

    return False


def detect_competitors_with_ai(client, responses_text: str, brand_name: str, max_competitors: int = 50) -> List[str]:
    """Use AI to detect competitors mentioned in the responses - FIXED VERSION"""

    # More specific prompt to catch ALL brand names including variations
    prompt = f"""You are analyzing AI responses about "{brand_name}". Extract ALL competitor brand names mentioned.

TEXT TO ANALYZE:
{responses_text}

TASK: Find ALL competitor brands/companies mentioned in the text above.

RULES:
1. Include ALL forms of brand names:
   - Full names (e.g., "Otter.ai", "Notion", "Evernote")
   - Short names (e.g., "Otter" when referring to Otter.ai)
   - With or without domains (e.g., both "Otter" and "Otter.ai")
   - With or without suffixes (e.g., both "MeetGeek" and "MeetGeek AI")

2. EXCLUDE ONLY:
   - "{brand_name}" and its direct products
   - Generic terms ("competitors", "other apps", "alternatives")
   - Non-brand descriptive words

3. Look for brands in:
   - Direct mentions ("Otter is the best...")
   - Lists ("apps like Otter, Notion, and...")
   - Comparisons ("better than Otter")
   - Any context where a brand is named

Return a JSON array of ALL unique brand names found.
If a brand appears in multiple forms (e.g., "Otter" and "Otter.ai"), include the most complete version.

Example: ["Otter.ai", "Notion", "Evernote", "Google Keep"]

IMPORTANT: Actually READ the text and extract brands that ARE THERE. Do not invent brands.

JSON array only:"""

    try:
        # Call the AI to analyze competitors
        if hasattr(client, 'client'):  # OpenAI client
            # Ensure we're using a good model
            model_to_use = client.model
            if 'gpt-3.5' in model_to_use:
                model_to_use = 'gpt-4o'  # Upgrade to GPT-4o for better detection

            print(f"=== COMPETITOR DETECTION DEBUG ===")
            print(f"Model used: {model_to_use}")
            print(f"Brand being analyzed: {brand_name}")
            print(f"Text length: {len(responses_text)} chars")
            print(f"First 200 chars: {responses_text[:200]}...")
            print("=================================")

            response = client.client.chat.completions.create(
                model=model_to_use,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2000
            )
            result = response.choices[0].message.content.strip()
        else:
            result = client.call_api(prompt)

        print(f"Raw AI response: {result[:200]}...")

        # Parse the JSON response
        try:
            # Clean up the response
            result = result.strip()
            if result.startswith('```json'):
                result = result[7:]
            if result.startswith('```'):
                result = result[3:]
            if result.endswith('```'):
                result = result[:-3]

            competitors = json.loads(result.strip())
            print(f"Competitors found by AI: {competitors}")

            if isinstance(competitors, list):
                valid_competitors = []
                seen_normalized = set()  # Track normalized versions

                for comp in competitors:
                    if isinstance(comp, str) and comp.strip():
                        cleaned = comp.strip()

                        # Skip if it's the brand itself
                        if is_same_brand(cleaned, brand_name) or contains_brand_name(cleaned, brand_name):
                            print(f"Skipping '{cleaned}' - same as brand or contains brand name")
                            continue

                        # Normalize for deduplication (e.g., "Otter" and "Otter.ai" -> "otter")
                        normalized = cleaned.lower().replace('.ai', '').replace('.com', '').replace(' ai', '')

                        # If we haven't seen this normalized form, add it
                        if normalized not in seen_normalized:
                            valid_competitors.append(cleaned)
                            seen_normalized.add(normalized)
                        else:
                            print(f"Skipping '{cleaned}' - duplicate of already found competitor")

                print(f"Final valid competitors: {valid_competitors}")
                return valid_competitors[:max_competitors]
            else:
                return []

        except json.JSONDecodeError as e:
            print(f"JSON parsing failed: {e}")
            print(f"Failed to parse: {result}")
            return extract_brands_from_text(responses_text, brand_name)

    except Exception as e:
        print(f"Error detecting competitors: {e}")
        return extract_brands_from_text(responses_text, brand_name)


def extract_brands_from_text(text: str, brand_name: str) -> List[str]:
    """Extract brand names from text using enhanced pattern matching"""

    brands = set()

    print(f"=== FALLBACK EXTRACTION ===")
    print(f"Extracting from text of length: {len(text)}")

    # First, look for specific app/service mentions
    app_patterns = [
        # "app like X", "apps like X, Y, Z"
        r'apps?\s+(?:like|such as|including)\s+([A-Za-z0-9\.\s,\-]+?)(?:\s+(?:and|or|are|is)|\.|\?|$)',

        # "X is the best", "X offers", etc.
        r'([A-Z][A-Za-z0-9\.]*(?:\s+[A-Z]?[A-Za-z0-9\.]*){0,2})\s+(?:is|are|offers?|provides?|uses?|has|features?)\s+',

        # "One of the best ... is X"
        r'(?:best|top|good|popular)\s+[\w\s]+\s+(?:is|are)\s+([A-Z][A-Za-z0-9\.]+)',

        # Specific pattern for "Otter" mentions
        r'\b(Otter(?:\.ai)?)\b',

        # Apps with .ai, .com, etc.
        r'([A-Z][A-Za-z0-9]+\.(?:ai|com|io|app))\b',
    ]

    for pattern in app_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            if ',' in match or ' and ' in match or ' or ' in match:
                # Split lists
                parts = re.split(r'[,\s]+(?:and|or)\s+|,\s*', match)
                for part in parts:
                    part = part.strip()
                    if part and not is_same_brand(part, brand_name) and not contains_brand_name(part, brand_name):
                        brands.add(part)
                        print(f"Found via app pattern: {part}")
            else:
                match = match.strip()
                if match and not is_same_brand(match, brand_name) and not contains_brand_name(match, brand_name):
                    brands.add(match)
                    print(f"Found via app pattern: {match}")

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
                'Based', 'Founded', 'Established', 'Created', 'Developed',
                'One', 'Two', 'Three', 'Many', 'Several', 'Few', 'All'
            }

            # Check if it's likely a brand - ENHANCED FILTERING
            if (cleaned and
                    cleaned not in non_brands and
                    not is_same_brand(cleaned, brand_name) and  # Use helper function
                    not contains_brand_name(cleaned, brand_name) and  # Use helper function
                    len(cleaned) > 2 and
                    not cleaned.isupper() and  # Avoid acronyms like "USA"
                    any(c.isupper() for c in cleaned)):  # Has at least one capital letter

                brands.add(cleaned)
                print(f"Found via pattern: {cleaned}")

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
            # Apply same filtering
            if not is_same_brand(match, brand_name) and not contains_brand_name(match, brand_name):
                brands.add(match)
                print(f"Found via special pattern: {match}")

    print(f"Total brands found by regex: {brands}")
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
        print(f"AI found only {len(detected_competitors)} competitors, trying fallback extraction...")
        fallback_competitors = extract_brands_from_text(responses_text, brand_name)

        # Merge with AI results, avoiding duplicates
        for comp in fallback_competitors:
            comp_normalized = comp.lower().replace('.ai', '').replace('.com', '')
            already_found = False

            for existing in detected_competitors:
                existing_normalized = existing.lower().replace('.ai', '').replace('.com', '')
                if comp_normalized == existing_normalized:
                    already_found = True
                    break

            if not already_found:
                detected_competitors.append(comp)
                print(f"Added from fallback: {comp}")

    # Additional filtering to ensure no brand variants slip through
    final_competitors = []
    for comp in detected_competitors:
        if not is_same_brand(comp, brand_name) and not contains_brand_name(comp, brand_name):
            final_competitors.append(comp)

    # Filter out existing competitors if provided
    if existing_competitors:
        existing_lower = [c.lower() for c in existing_competitors]
        new_suggestions = [c for c in final_competitors if c.lower() not in existing_lower]
    else:
        new_suggestions = final_competitors

    print(f"=== FINAL RESULTS ===")
    print(f"Total detected: {len(final_competitors)}")
    print(f"Final competitors: {final_competitors}")
    print("====================")

    return {
        'detected': final_competitors,
        'new_suggestions': new_suggestions
    }