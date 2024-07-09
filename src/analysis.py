import os
import json
import requests
from bs4 import BeautifulSoup
import tldextract
from typing import List, Optional, Dict, Any
from typing_extensions import TypedDict
import google.generativeai as genai
import logging
import re
import unicodedata

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configure the Gemini API
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

class Category(TypedDict):
    name: str
    user_friendly_aspect: str
    concerning_aspect: str
    score: float
    justification: str

class TosAnalysis(TypedDict):
    initial_assessment: str
    categories: List[Category]
    final_score: float
    letter_grade: str
    summary: str
    green_flags: List[str]
    red_flags: List[str]

def fetch_tos_document(url: str) -> Optional[str]:
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        tos_text = soup.get_text()
        if not tos_text.strip():
            logger.warning("Fetched ToS document is empty")
            return None
        return tos_text
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching ToS document: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error while fetching ToS document: {e}")
        return None

def extract_company_name(url: str) -> str:
    ext = tldextract.extract(url)
    domain = ext.registered_domain
    company_name = domain.split('.')[0].capitalize()
    return company_name

def generate_tos_analysis_prompt(company_name: str, tos_text: str) -> str:
    return f"""Analyze the Terms of Service (ToS) for {company_name} objectively and comprehensively. Your analysis should be balanced, considering both user-friendly and potentially concerning aspects.

Initial Assessment:
1. Identify the 3 most notable aspects of this ToS, whether positive or negative. (50 words max)

For each category below, provide:
a) Most user-friendly aspect (1 sentence)
b) Most concerning aspect (1 sentence)
c) Score (-2 to +2)
d) Brief justification (30 words max)

Scoring Guide:
0-2: Highly concerning or user-unfriendly
3-4: Somewhat concerning or below average
5: Neutral or industry standard
6-7: User-friendly or above average
8-10: Exceptionally user-friendly or protective of user rights

Categories:
1. Clarity and Readability: How easy is the ToS to understand for an average user?
2. Privacy and Data Security: How well does it protect user data and privacy?
3. Data Collection and Usage: How transparent and fair are data practices?
4. User Rights and Control: What level of control do users have over their data and account?
5. Liability and Disclaimers: How balanced are the liability terms between user and company?
6. Termination and Account Suspension: How fair and clear are these processes?
7. Changes to Terms: How are users notified and what rights do they have regarding changes?

Overall Assessment:
1. Calculate the final score: Convert category scores to 0-10 scale, then take the weighted average using category weights [15%, 25%, 20%, 15%, 10%, 5%, 5%]. Round to one decimal place.
2. Assign a letter grade based on the final score:
   9.0-10: A+ | 8.5-8.9: A | 8.0-8.4: A- | 7.5-7.9: B+ | 7.0-7.4: B | 6.5-6.9: B-
   6.0-6.4: C+ | 5.5-5.9: C | 5.0-5.4: C- | 4.5-4.9: D+ | 4.0-4.4: D | 3.5-3.9: D-
   0-3.4: F
3. Summarize the ToS, highlighting the most significant positive and negative aspects. (50 words max)
4. List up to 3 green flags (user-friendly practices) and 3 red flags (concerning practices), if any.

Terms of Service to analyze:
{tos_text[:100000]}  # Limit input to approximately 100,000 tokens
"""

def analyze_tos(tos_text: str, company_name: str) -> Dict[str, Any]:
    if not tos_text:
        return {"error": "Unable to fetch the Terms of Service document."}
    
    if not os.environ.get("GEMINI_API_KEY"):
        return {"error": "GEMINI_API_KEY environment variable is not set."}

    try:
        model = genai.GenerativeModel('gemini-1.5-pro-latest')
        
        prompt = generate_tos_analysis_prompt(company_name, tos_text)

        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.2,
                top_p=1,
                top_k=32,
                max_output_tokens=2048,
                response_mime_type="application/json"
            )
        )
        
        logger.debug(f"Raw API response: {response}")

        # Extract the text content from the response
        try:
            response_text = response.candidates[0].content.parts[0].text
        except (AttributeError, IndexError) as e:
            logger.error(f"Error extracting text from response: {e}")
            logger.debug(f"Response structure: {response}")
            return {"error": "Unexpected response format from Gemini API"}
        
        logger.debug(f"Response text: {response_text}")

        # Parse the JSON response
        analysis_json = parse_and_clean_json(response_text)

        # Post-processing to ensure consistency
        try:
            cleaned_analysis = post_process_analysis(analysis_json)
            cleaned_analysis['company_name'] = company_name  # Add company name to the analysis result
            return cleaned_analysis
        except ValueError as ve:
            logger.error(f"Error in post-processing: {ve}")
            return {
                "error": f"Error in post-processing: {str(ve)}",
                "raw_response": response_text[:1000]
            }

    except genai.types.generation_types.BlockedPromptException as e:
        logger.error(f"Blocked prompt exception: {e}")
        return {"error": "The analysis request was blocked due to content restrictions."}
    except Exception as e:
        logger.exception("An error occurred while analyzing the Terms of Service")
        return {"error": f"An error occurred while analyzing the Terms of Service: {str(e)}"}

def post_process_analysis(analysis: Dict[str, Any]) -> Dict[str, Any]:
    # Check if the analysis is a dictionary
    if not isinstance(analysis, dict):
        raise ValueError(f"Expected dictionary, got {type(analysis)}")
    
    # Ensure all required fields are present
    required_keys = ['initial_assessment', 'categories', 'final_score', 'letter_grade', 'summary', 'green_flags', 'red_flags']
    for field in required_keys:
        if field not in analysis:
            analysis[field] = "Not provided" if field in ['initial_assessment', 'letter_grade', 'summary'] else []

    # Ensure final_score is within 0-10 range and is a float
    try:
        final_score = analysis.get('final_score', 0)
        if isinstance(final_score, list):
            final_score = final_score[0] if final_score else 0
        analysis['final_score'] = max(0, min(10, float(final_score)))
    except (ValueError, TypeError):
        analysis['final_score'] = 0

    # Ensure categories have all required fields
    if isinstance(analysis['categories'], list):
        for category in analysis['categories']:
            required_category_fields = ['name', 'user_friendly_aspect', 'concerning_aspect', 'score', 'justification']
            for field in required_category_fields:
                if field not in category:
                    category[field] = "Not provided" if field != 'score' else 0

            # Ensure category score is within 0 to 10 range and is a float
            try:
                category['score'] = max(0, min(10, float(category.get('score', 0))))
            except ValueError:
                category['score'] = 0
    else:
        analysis['categories'] = []

    # Ensure green_flags and red_flags are lists
    for flag_type in ['green_flags', 'red_flags']:
        if not isinstance(analysis[flag_type], list):
            analysis[flag_type] = [analysis[flag_type]] if analysis[flag_type] else []

    # Ensure summary is a string
    analysis['summary'] = str(analysis.get('summary', ''))

    return analysis

def parse_and_clean_json(response_text: str) -> Dict[str, Any]:
    """
    Attempt to parse JSON, clean, and restructure it to match expected format.
    """
    try:
        # First attempt to parse the JSON as-is
        parsed_json = json.loads(response_text)
    except json.JSONDecodeError:
        # If parsing fails, apply cleaning steps
        cleaned_text = clean_json_response(response_text)
        try:
            # Attempt to parse the cleaned JSON
            parsed_json = json.loads(cleaned_text)
        except json.JSONDecodeError as e:
            # If it still fails, attempt to extract structured data
            logger.error(f"Failed to parse cleaned JSON: {e}")
            logger.error(f"Cleaned text: {cleaned_text[:500]}")
            parsed_json = extract_structured_data(cleaned_text)
            if not parsed_json:
                return {
                    "error": "Failed to parse the API response as JSON, even after cleaning.",
                    "raw_response": response_text[:1000]
                }

    # Restructure the parsed JSON to match expected format
    restructured_json = {
        "initial_assessment": parsed_json.get("Initial Assessment", ""),
        "categories": [],
        "final_score": 0,
        "letter_grade": "",
        "summary": "",
        "green_flags": [],
        "red_flags": []
    }

    # Extract Overall Assessment data
    overall_assessment = parsed_json.get("Overall Assessment", {})
    if isinstance(overall_assessment, dict):
        restructured_json["final_score"] = float(overall_assessment.get("Final Score", 0))
        restructured_json["letter_grade"] = overall_assessment.get("Letter Grade", "")
        restructured_json["summary"] = overall_assessment.get("Summary", "")
        restructured_json["green_flags"] = overall_assessment.get("Green Flags", [])
        restructured_json["red_flags"] = overall_assessment.get("Red Flags", [])

    # Convert category data into list format
    category_names = ["Clarity and Readability", "Privacy and Data Security", "Data Collection and Usage",
                      "User Rights and Control", "Liability and Disclaimers", "Termination and Account Suspension",
                      "Changes to Terms"]
    
    for name in category_names:
        if name in parsed_json:
            category = parsed_json[name]
            restructured_json["categories"].append({
                "name": name,
                "user_friendly_aspect": category.get("a", ""),
                "concerning_aspect": category.get("b", ""),
                "score": float(category.get("c", 0)),
                "justification": category.get("d", "")
            })

    return restructured_json

def clean_json_response(response_text: str) -> str:
    """
    Apply a series of cleaning steps to the JSON response text.
    """
    # Remove any leading or trailing whitespace
    response_text = response_text.strip()
    
    # Ensure the response starts and ends with curly braces
    if not response_text.startswith('{'):
        response_text = '{' + response_text
    if not response_text.endswith('}'):
        response_text = response_text + '}'
    
    # Replace single quotes with double quotes, but not within words (like apostrophes)
    response_text = re.sub(r"(?<!\w)'(?!\w)", '"', response_text)
    
    # Handle newlines within string values
    response_text = re.sub(r'(?<!\\)\\n', r'\\n', response_text)
    
    # Remove any control characters
    response_text = ''.join(ch for ch in response_text if unicodedata.category(ch)[0] != 'C')
    
    # Escape unescaped quotes within string values
    response_text = re.sub(r'(?<!\\)"(?=(?:(?:[^"]*"){2})*[^"]*$)', r'\"', response_text)
    
    # Handle escaped apostrophes
    response_text = response_text.replace("\\'", "'")
    
    # Replace double backslashes with single backslashes
    response_text = response_text.replace("\\\\", "\\")
    
    # Ensure all keys are properly quoted
    response_text = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', response_text)
    
    # Remove any trailing commas in objects or arrays
    response_text = re.sub(r',\s*([\]}])', r'\1', response_text)
    
    return response_text

def main():
    tos_url = input("Enter the URL of the Terms of Service document: ")
    tos_text = fetch_tos_document(tos_url)
    
    if tos_text:
        company_name = extract_company_name(tos_url)
        print(f"Analyzing Terms of Service for {company_name}...\n")
        
        analysis = analyze_tos(tos_text, company_name)
        print(json.dumps(analysis, indent=2))
    else:
        print("Unable to fetch the Terms of Service document.")

if __name__ == "__main__":
    main()
def extract_structured_data(text: str) -> Dict[str, Any]:
    """
    Attempt to extract structured data from the text when JSON parsing fails.
    """
    data = {}
    
    # Extract Initial Assessment
    initial_assessment_match = re.search(r'"Initial Assessment":\s*"([^"]*)"', text)
    if initial_assessment_match:
        data['Initial Assessment'] = initial_assessment_match.group(1)
    
    # Extract categories
    categories = ["Clarity and Readability", "Privacy and Data Security", "Data Collection and Usage",
                  "User Rights and Control", "Liability and Disclaimers", "Termination and Account Suspension",
                  "Changes to Terms"]
    for category in categories:
        category_match = re.search(rf'"{category}":\s*{{([^}}]*)}}', text)
        if category_match:
            category_data = {}
            category_content = category_match.group(1)
            for key in ['a', 'b', 'c', 'd']:
                key_match = re.search(rf'"{key}":\s*"([^"]*)"', category_content)
                if key_match:
                    category_data[key] = key_match.group(1)
            data[category] = category_data
    
    # Extract Overall Assessment
    overall_assessment_match = re.search(r'"Overall Assessment":\s*{{([^}}]*)}}', text)
    if overall_assessment_match:
        overall_content = overall_assessment_match.group(1)
        data['Overall Assessment'] = {}
        for key in ['Final Score', 'Letter Grade', 'Summary', 'Green Flags', 'Red Flags']:
            key_match = re.search(rf'"{key}":\s*"?([^",}}]*)"?', overall_content)
            if key_match:
                data['Overall Assessment'][key] = key_match.group(1)
    
    return data
