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
from html import unescape
from requests.exceptions import Timeout, RequestException

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
    company_name: str  # Add this field to match the expected structure

class TosAnalysis(TypedDict):
    initial_assessment: str
    categories: List[Category]
    final_score: float
    letter_grade: str
    summary: str
    green_flags: List[str]
    red_flags: List[str]
    company_name: str  # Add this field to match the expected structure

def fetch_tos_document(url: str) -> Optional[str]:
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)  # Increased timeout
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        tos_text = soup.get_text()
        tos_text = unescape(tos_text)  # Unescape HTML entities
        tos_text = re.sub(r'\s+', ' ', tos_text).strip()  # Normalize whitespace
        if not tos_text:
            logger.warning("Fetched ToS document is empty")
            return None
        return tos_text[:500000]  # Limit to 500,000 characters
    except Timeout:
        logger.error("Timeout error while fetching ToS document")
        return None
    except RequestException as e:
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
    return f"""Analyze the Terms of Service (ToS) for {company_name} objectively and comprehensively. Your analysis should be balanced, considering both user-friendly and potentially concerning aspects. Provide your response as a JSON object with the following structure:

{{
    "initial_assessment": "Identify the 3 most notable aspects of this ToS, whether positive or negative. (50 words max)",
    "categories": [
        {{
            "name": "Category Name",
            "user_friendly_aspect": "Most user-friendly aspect (1 sentence)",
            "concerning_aspect": "Most concerning aspect (1 sentence)",
            "score": 0,
            "justification": "Brief justification (30 words max)"
        }},
        // Repeat for all 7 categories
    ],
    "final_score": 0.0,
    "letter_grade": "Letter grade based on the final score",
    "summary": "Summarize the ToS, highlighting the most significant positive and negative aspects. (50 words max)",
    "green_flags": ["List up to 3 user-friendly practices"],
    "red_flags": ["List up to 3 concerning practices"]
}}

Categories to analyze:
1. Clarity and Readability
2. Privacy and Data Security
3. Data Collection and Usage
4. User Rights and Control
5. Liability and Disclaimers
6. Termination and Account Suspension
7. Changes to Terms

Scoring Guide:
0-2: Highly concerning or user-unfriendly
3-4: Somewhat concerning or below average
5: Neutral or industry standard
6-7: User-friendly or above average
8-10: Exceptionally user-friendly or protective of user rights

Letter Grade Guide:
9.0-10: A+ | 8.5-8.9: A | 8.0-8.4: A- | 7.5-7.9: B+ | 7.0-7.4: B | 6.5-6.9: B-
6.0-6.4: C+ | 5.5-5.9: C | 5.0-5.4: C- | 4.5-4.9: D+ | 4.0-4.4: D | 3.5-3.9: D-
0-3.4: F

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
                response_mime_type="application/json",
                response_schema=TosAnalysis
            )
        )
        
        logger.debug(f"Raw API response: {response}")

        try:
            analysis_result = json.loads(response.text)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON response: {e}")
            return {"error": "Unable to parse the JSON response from Gemini API"}

        # Post-processing to ensure consistency
        try:
            cleaned_analysis = post_process_analysis(analysis_result)
            cleaned_analysis['company_name'] = company_name  # Add company name to the analysis result
            return cleaned_analysis
        except ValueError as ve:
            logger.error(f"Error in post-processing: {ve}")
            return {
                "error": f"Error in post-processing: {str(ve)}",
                "raw_response": response.text[:1000]
            }

    except genai.types.generation_types.BlockedPromptException as e:
        logger.error(f"Blocked prompt exception: {e}")
        return {"error": "The analysis request was blocked due to content restrictions."}
    except Timeout:
        logger.error("Timeout error while calling Gemini API")
        return {"error": "The analysis request timed out. Please try again later."}
    except Exception as e:
        logger.exception("An error occurred while analyzing the Terms of Service")
        return {"error": f"An error occurred while analyzing the Terms of Service: {str(e)}"}

def post_process_analysis(analysis: Dict[str, Any]) -> Dict[str, Any]:
    # Check if the analysis is a dictionary
    if not isinstance(analysis, dict):
        raise ValueError(f"Expected dictionary, got {type(analysis)}")
    
    # Extract initial assessment
    initial_assessment = analysis.get('Initial Assessment', {})
    if isinstance(initial_assessment, dict):
        analysis['initial_assessment'] = initial_assessment.get('notable_aspects', [])
    else:
        analysis['initial_assessment'] = []

    # Extract categories
    categories = []
    category_names = [
        "Clarity and Readability", "Privacy and Data Security", "Data Collection and Usage",
        "User Rights and Control", "Liability and Disclaimers", "Termination and Account Suspension",
        "Changes to Terms"
    ]
    for name in category_names:
        category = analysis.get(name, {})
        if isinstance(category, dict):
            categories.append({
                'name': name,
                'user_friendly_aspect': category.get('user_friendly_aspect', ''),
                'concerning_aspect': category.get('concerning_aspect', ''),
                'score': float(category.get('score', 0)),
                'justification': category.get('justification', '')
            })
    analysis['categories'] = categories

    # Extract overall assessment
    overall_assessment = analysis.get('Overall Assessment', {})
    if isinstance(overall_assessment, dict):
        analysis['final_score'] = float(overall_assessment.get('final_score', 0))
        analysis['letter_grade'] = overall_assessment.get('letter_grade', '')
        analysis['summary'] = overall_assessment.get('summary', '')
        analysis['green_flags'] = overall_assessment.get('green_flags', [])
        analysis['red_flags'] = overall_assessment.get('red_flags', [])
    else:
        analysis['final_score'] = 0
        analysis['letter_grade'] = ''
        analysis['summary'] = ''
        analysis['green_flags'] = []
        analysis['red_flags'] = []

    # Ensure all required fields are present and properly formatted
    analysis['final_score'] = max(0, min(10, float(analysis['final_score'])))
    analysis['letter_grade'] = str(analysis['letter_grade'])
    analysis['summary'] = str(analysis['summary'])
    analysis['green_flags'] = list(analysis['green_flags'])
    analysis['red_flags'] = list(analysis['red_flags'])

    return analysis

def parse_structured_response(response_text: str) -> Dict[str, Any]:
    sections = response_text.split('\n\n')
    parsed_data = {}

    for section in sections:
        if section.startswith('INITIAL_ASSESSMENT'):
            parsed_data['initial_assessment'] = section.replace('INITIAL_ASSESSMENT\n', '').strip()
        elif section.startswith('CATEGORIES'):
            categories = []
            current_category = {}
            for line in section.split('\n')[1:]:
                if line.startswith('Category:'):
                    if current_category:
                        categories.append(current_category)
                        current_category = {}
                    current_category['name'] = line.split(': ', 1)[1]
                elif line.startswith('User-friendly aspect:'):
                    current_category['user_friendly_aspect'] = line.split(': ', 1)[1]
                elif line.startswith('Concerning aspect:'):
                    current_category['concerning_aspect'] = line.split(': ', 1)[1]
                elif line.startswith('Score:'):
                    current_category['score'] = float(line.split(': ', 1)[1])
                elif line.startswith('Justification:'):
                    current_category['justification'] = line.split(': ', 1)[1]
            if current_category:
                categories.append(current_category)
            parsed_data['categories'] = categories
        elif section.startswith('OVERALL_ASSESSMENT'):
            for line in section.split('\n')[1:]:
                if line.startswith('Final Score:'):
                    parsed_data['final_score'] = float(line.split(': ', 1)[1])
                elif line.startswith('Letter Grade:'):
                    parsed_data['letter_grade'] = line.split(': ', 1)[1]
                elif line.startswith('Summary:'):
                    parsed_data['summary'] = line.split(': ', 1)[1]
                elif line.startswith('Green Flags:'):
                    parsed_data['green_flags'] = [flag.strip() for flag in line.split(':', 1)[1].split(',')]
                elif line.startswith('Red Flags:'):
                    parsed_data['red_flags'] = [flag.strip() for flag in line.split(':', 1)[1].split(',')]

    return parsed_data

def parse_and_clean_json(response_text: str) -> Dict[str, Any]:
    """
    Attempt to parse JSON, clean, and restructure it to match expected format.
    """
    def safe_float(value, default=0.0):
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

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
        "initial_assessment": str(parsed_json.get("Initial Assessment", "")),
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
        restructured_json["final_score"] = safe_float(overall_assessment.get("Final Score", 0))
        restructured_json["letter_grade"] = str(overall_assessment.get("Letter Grade", ""))
        restructured_json["summary"] = str(overall_assessment.get("Summary", ""))
        restructured_json["green_flags"] = list(overall_assessment.get("Green Flags", []))
        restructured_json["red_flags"] = list(overall_assessment.get("Red Flags", []))

    # Convert category data into list format
    category_names = ["Clarity and Readability", "Privacy and Data Security", "Data Collection and Usage",
                      "User Rights and Control", "Liability and Disclaimers", "Termination and Account Suspension",
                      "Changes to Terms"]
    
    categories = parsed_json.get("Categories", [])
    if isinstance(categories, list):
        for category in categories:
            for name in category_names:
                if name in category:
                    cat_data = category[name]
                    restructured_json["categories"].append({
                        "name": name,
                        "user_friendly_aspect": str(cat_data.get("a", "")),
                        "concerning_aspect": str(cat_data.get("b", "")),
                        "score": safe_float(cat_data.get("c", 0)),
                        "justification": str(cat_data.get("d", ""))
                    })
    elif isinstance(categories, dict):
        for name in category_names:
            if name in categories:
                cat_data = categories[name]
                restructured_json["categories"].append({
                    "name": name,
                    "user_friendly_aspect": str(cat_data.get("a", "")),
                    "concerning_aspect": str(cat_data.get("b", "")),
                    "score": safe_float(cat_data.get("c", 0)),
                    "justification": str(cat_data.get("d", ""))
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
    
    # Handle escaped quotes within string values
    response_text = re.sub(r'(?<!\\)\\(?!\\)"', '\\"', response_text)
    
    # Replace single quotes with double quotes, but not within words (like apostrophes)
    response_text = re.sub(r"(?<!\w)'(?!\w)", '"', response_text)
    
    # Handle newlines within string values
    response_text = re.sub(r'(?<!\\)\\n', r'\\n', response_text)
    
    # Remove any control characters
    response_text = ''.join(ch for ch in response_text if unicodedata.category(ch)[0] != 'C')
    
    # Ensure all keys are properly quoted
    response_text = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', response_text)
    
    # Remove any trailing commas in objects or arrays
    response_text = re.sub(r',\s*([\]}])', r'\1', response_text)
    
    # Handle potential nested JSON strings
    def replace_nested_json(match):
        try:
            return json.dumps(json.loads(match.group(1)))
        except json.JSONDecodeError:
            return match.group(0)
    
    response_text = re.sub(r'"(\{[^}]*\})"', replace_nested_json, response_text)
    
    # Handle truncated JSON
    try:
        json.loads(response_text)
    except json.JSONDecodeError as e:
        # If the error is due to truncation, attempt to close the JSON structure
        if "Expecting ',' delimiter" in str(e):
            # Find the last complete object or array
            last_complete = max(response_text.rfind('}'), response_text.rfind(']'))
            if last_complete != -1:
                response_text = response_text[:last_complete+1]
                # Close any open brackets or braces
                open_brackets = response_text.count('{') - response_text.count('}')
                open_squares = response_text.count('[') - response_text.count(']')
                response_text += '}' * open_brackets + ']' * open_squares
    
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
