import os
import json
import requests
from bs4 import BeautifulSoup
import tldextract
from typing import List, Optional, Dict, Any
from typing_extensions import TypedDict
import google.generativeai as genai
import logging

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
        return tos_text
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching ToS document: {e}")
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
-2: Highly concerning or user-unfriendly
-1: Somewhat concerning or below average
 0: Neutral or industry standard
+1: User-friendly or above average
+2: Exceptionally user-friendly or protective of user rights

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

Provide your analysis in the following JSON format:
{{
  "initial_assessment": "string",
  "categories": [
    {{
      "name": "string",
      "user_friendly_aspect": "string",
      "concerning_aspect": "string",
      "score": number,
      "justification": "string"
    }}
  ],
  "final_score": number,
  "letter_grade": "string",
  "summary": "string",
  "green_flags": ["string"],
  "red_flags": ["string"]
}}

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
            generation_config={
                "temperature": 0.2,
                "top_p": 1,
                "top_k": 32,
                "max_output_tokens": 2048,
            }
        )
        
        logger.debug(f"Raw API response: {response}")

        # Extract the text content from the response
        if hasattr(response, 'parts'):
            response_text = response.parts[0].text
        elif hasattr(response, 'text'):
            response_text = response.text
        else:
            response_text = str(response)

        # Clean the response text by removing markdown code block syntax
        cleaned_response = clean_response_text(response_text)
        
        logger.debug(f"Cleaned response text: {cleaned_response[:1000]}")  # Log first 1000 characters

        # Parse the JSON response
        try:
            analysis_json = json.loads(cleaned_response)
            
            # Post-processing to ensure consistency
            cleaned_analysis = post_process_analysis(analysis_json)

            return cleaned_analysis

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse API response as JSON: {e}")
            logger.error(f"Problematic text: {cleaned_response[:500]}")  # Log the first 500 characters of the problematic text
            return {
                "error": "Failed to parse the API response as JSON.",
                "raw_response": cleaned_response[:1000]  # Include first 1000 characters of the cleaned response
            }

    except Exception as e:
        logger.exception("An error occurred while analyzing the Terms of Service")
        return {"error": f"An error occurred while analyzing the Terms of Service: {str(e)}"}

def clean_response_text(text: str) -> str:
    """Remove markdown code block syntax and trim whitespace."""
    # Remove ```json from the start and ``` from the end if present
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()

def post_process_analysis(analysis: Dict[str, Any]) -> Dict[str, Any]:
    # Ensure all required fields are present
    required_fields = ['initial_assessment', 'categories', 'final_score', 'letter_grade', 'summary', 'green_flags', 'red_flags']
    for field in required_fields:
        if field not in analysis:
            analysis[field] = "Not provided" if field in ['initial_assessment', 'letter_grade', 'summary'] else []

    # Ensure final_score is within 0-10 range
    analysis['final_score'] = max(0, min(10, analysis['final_score']))

    # Ensure categories have all required fields
    for category in analysis['categories']:
        required_category_fields = ['name', 'user_friendly_aspect', 'concerning_aspect', 'score', 'justification']
        for field in required_category_fields:
            if field not in category:
                category[field] = "Not provided" if field != 'score' else 0

        # Ensure category score is within -2 to 2 range
        category['score'] = max(-2, min(2, category['score']))

    # Ensure green_flags and red_flags are lists
    for flag_type in ['green_flags', 'red_flags']:
        if not isinstance(analysis[flag_type], list):
            analysis[flag_type] = [analysis[flag_type]] if analysis[flag_type] else []

    return analysis

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