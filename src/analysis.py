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
    company_name: str

def fetch_tos_document(url: str) -> Optional[str]:
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
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
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        prompt = generate_tos_analysis_prompt(company_name, tos_text)

        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.2,
                top_p=1,
                top_k=32,
                max_output_tokens=2048,
                response_mime_type="application/json"
                # Remove response_schema as it's causing issues
            )
        )
        
        logger.debug(f"Raw API response: {response}")

        # Check if the response has content
        if not response.candidates or not response.candidates[0].content:
            raise ValueError("No content in the API response")

        # Extract the text content from the response
        response_text = response.candidates[0].content.parts[0].text
        
        # Parse the JSON content
        analysis_result = json.loads(response_text)

        # Post-processing to ensure consistency
        cleaned_analysis = post_process_analysis(analysis_result)
        cleaned_analysis['company_name'] = company_name

        return cleaned_analysis

    except genai.types.generation_types.BlockedPromptException as e:
        logger.error(f"Blocked prompt exception: {e}")
        return {"error": "The analysis request was blocked due to content restrictions."}
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON response: {e}")
        return {"error": "Unable to parse the JSON response from Gemini API"}
    except Exception as e:
        logger.exception("An error occurred while analyzing the Terms of Service")
        return {"error": f"An error occurred while analyzing the Terms of Service: {str(e)}"}

def post_process_analysis(analysis: Dict[str, Any]) -> Dict[str, Any]:
    # Ensure all required fields are present and properly formatted
    analysis['initial_assessment'] = str(analysis.get('initial_assessment', ''))
    analysis['final_score'] = max(0, min(10, float(analysis.get('final_score', 0))))
    analysis['letter_grade'] = str(analysis.get('letter_grade', ''))
    analysis['summary'] = str(analysis.get('summary', ''))
    analysis['green_flags'] = list(analysis.get('green_flags', []))
    analysis['red_flags'] = list(analysis.get('red_flags', []))

    # Ensure categories are properly structured
    categories = []
    for category in analysis.get('categories', []):
        if isinstance(category, dict):
            categories.append({
                'name': str(category.get('name', '')),
                'user_friendly_aspect': str(category.get('user_friendly_aspect', '')),
                'concerning_aspect': str(category.get('concerning_aspect', '')),
                'score': max(0, min(10, float(category.get('score', 0)))),
                'justification': str(category.get('justification', ''))
            })
    analysis['categories'] = categories

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
        "initial_assessment": str(parsed_json.get("initial_assessment", "")),
        "categories": [],
        "final_score": 0,
        "letter_grade": "",
        "summary": "",
        "green_flags": [],
        "red_flags": []
    }

    # Extract Overall Assessment data
    overall_assessment = parsed_json.get("overall_assessment", {})
    if isinstance(overall_assessment, dict):
        restructured_json["final_score"] = safe_float(overall_assessment.get("final_score", 0))
        restructured_json["letter_grade"] = str(overall_assessment.get("letter_grade", ""))
        restructured_json["summary"] = str(overall_assessment.get("summary", ""))
        restructured_json["green_flags"] = list(overall_assessment.get("green_flags", []))
        restructured_json["red_flags"] = list(overall_assessment.get("red_flags", []))

    # Convert category data into list format
    category_names = ["Clarity and Readability", "Privacy and Data Security", "Data Collection and Usage",
                      "User Rights and Control", "Liability and Disclaimers", "Termination and Account Suspension",
                      "Changes to Terms"]
    
    categories = parsed_json.get("categories", [])
    if isinstance(categories, list):
        for category in categories:
            if isinstance(category, dict):
                restructured_json["categories"].append({
                    "name": str(category.get("name", "")),
                    "user_friendly_aspect": str(category.get("user_friendly_aspect", "")),
                    "concerning_aspect": str(category.get("concerning_aspect", "")),
                    "score": safe_float(category.get("score", 0)),
                    "justification": str(category.get("justification", ""))
                })
    elif isinstance(categories, dict):
        for name in category_names:
            if name in categories:
                cat_data = categories[name]
                restructured_json["categories"].append({
                    "name": name,
                    "user_friendly_aspect": str(cat_data.get("user_friendly_aspect", "")),
                    "concerning_aspect": str(cat_data.get("concerning_aspect", "")),
                    "score": safe_float(cat_data.get("score", 0)),
                    "justification": str(cat_data.get("justification", ""))
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

def extract_structured_data(text: str) -> Dict[str, Any]:
    """
    Attempt to extract structured data from the text when JSON parsing fails.
    """
    data = {}
    
    # Extract Initial Assessment
    initial_assessment_match = re.search(r'"initial_assessment":\s*"([^"]*)"', text, re.IGNORECASE)
    if initial_assessment_match:
        data['initial_assessment'] = initial_assessment_match.group(1)
    
    # Extract categories
    categories = []
    category_pattern = r'"name":\s*"([^"]*)".*?"user_friendly_aspect":\s*"([^"]*)".*?"concerning_aspect":\s*"([^"]*)".*?"score":\s*([\d.]+).*?"justification":\s*"([^"]*)"'
    for match in re.finditer(category_pattern, text, re.DOTALL):
        categories.append({
            "name": match.group(1),
            "user_friendly_aspect": match.group(2),
            "concerning_aspect": match.group(3),
            "score": float(match.group(4)),
            "justification": match.group(5)
        })
    data['categories'] = categories
    
    # Extract overall assessment data
    final_score_match = re.search(r'"final_score":\s*([\d.]+)', text)
    if final_score_match:
        data['final_score'] = float(final_score_match.group(1))
    
    letter_grade_match = re.search(r'"letter_grade":\s*"([^"]*)"', text)
    if letter_grade_match:
        data['letter_grade'] = letter_grade_match.group(1)
    
    summary_match = re.search(r'"summary":\s*"([^"]*)"', text)
    if summary_match:
        data['summary'] = summary_match.group(1)
    
    green_flags_match = re.search(r'"green_flags":\s*\[(.*?)\]', text, re.DOTALL)
    if green_flags_match:
        data['green_flags'] = [flag.strip().strip('"') for flag in green_flags_match.group(1).split(',')]
    
    red_flags_match = re.search(r'"red_flags":\s*\[(.*?)\]', text, re.DOTALL)
    if red_flags_match:
        data['red_flags'] = [flag.strip().strip('"') for flag in red_flags_match.group(1).split(',')]
    
    return data

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