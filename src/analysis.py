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
    weight: int
    score: float
    explanation: str

class TosAnalysis(TypedDict):
    categories: List[Category]
    overall_score: float
    summary: str

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


def analyze_tos(tos_text: str, company_name: str) -> Dict[str, Any]:
    if not tos_text:
        return {"error": "Unable to fetch the Terms of Service document."}
    
    if not os.environ.get("GEMINI_API_KEY"):
        return {"error": "GEMINI_API_KEY environment variable is not set."}

    try:
        model = genai.GenerativeModel('gemini-1.5-pro-latest')
        
        prompt = f"""
        Analyze the following Terms of Service for {company_name} with a highly critical perspective. Be especially harsh on practices that violate user privacy, involve excessive data collection, or unfair data selling practices. Start from a baseline score of 5/10 for each category and adjust downwards for concerning practices. Provide a JSON output with the following structure:
        {{
            "categories": [
                {{
                    "name": "Clarity and Readability",
                    "weight": 15,
                    "score": 5,
                    "explanation": "Detailed critical explanation here"
                }},
                {{
                    "name": "Privacy and Data Security",
                    "weight": 25,
                    "score": 5,
                    "explanation": "Detailed critical explanation here"
                }},
                {{
                    "name": "Data Collection and Usage",
                    "weight": 20,
                    "score": 5,
                    "explanation": "Detailed critical explanation here"
                }},
                {{
                    "name": "User Rights and Control",
                    "weight": 15,
                    "score": 5,
                    "explanation": "Detailed critical explanation here"
                }},
                {{
                    "name": "Liability and Disclaimers",
                    "weight": 10,
                    "score": 5,
                    "explanation": "Detailed critical explanation here"
                }},
                {{
                    "name": "Termination and Account Suspension",
                    "weight": 5,
                    "score": 5,
                    "explanation": "Detailed critical explanation here"
                }},
                {{
                    "name": "Changes to Terms",
                    "weight": 5,
                    "score": 5,
                    "explanation": "Detailed critical explanation here"
                }},
                {{
                    "name": "Overall Fairness",
                    "weight": 5,
                    "score": 5,
                    "explanation": "Detailed critical explanation here"
                }}
            ],
            "overall_score": 5,
            "summary": "Highly critical overall summary of the analysis",
            "red_flags": ["List of significant concerns or red flags identified"]
        }}

        Guidelines for harsh criticism:
        1. Privacy and Data Security: Severely penalize any vague language about data usage, sharing with third parties, or lack of encryption standards.
        2. Data Collection and Usage: Harshly criticize any excessive data collection, unclear retention policies, or data selling practices.
        3. User Rights and Control: Severely dock points for any terms that limit user control over their data or make it difficult to delete accounts.
        4. Clarity and Readability: Be critical of any legal jargon or unnecessarily complex language.
        5. Overall Fairness: Significantly lower scores for any terms that seem to heavily favor the company over users.

        Provide detailed explanations for each category score, highlighting specific concerning clauses or practices.
        Calculate the overall_score as a weighted average of the category scores, but ensure it does not exceed 7/10 even if individual categories score higher.
        In the red_flags list, include any practices that are particularly alarming or user-unfriendly.

        Terms of Service to analyze:
        {tos_text[:100000]}  # Limit the input to approximately 100,000 tokens
        """

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
            
            # Post-processing to ensure harshness
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
    # Ensure overall score doesn't exceed 7
    analysis['overall_score'] = min(analysis['overall_score'], 7.0)

    # Apply additional penalties for critical issues
    critical_categories = ['Privacy and Data Security', 'Data Collection and Usage', 'User Rights and Control']
    for category in analysis['categories']:
        if category['name'] in critical_categories and category['score'] < 3:
            analysis['overall_score'] *= 0.9  # 10% penalty for each critical category scoring below 3

    # Cap the overall score if any category is extremely low
    if any(category['score'] < 2 for category in analysis['categories']):
        analysis['overall_score'] = min(analysis['overall_score'], 5.0)

    # Ensure red flags are prominently featured
    if 'red_flags' not in analysis or not analysis['red_flags']:
        analysis['red_flags'] = ["No specific red flags identified, but this does not guarantee the absence of concerning practices."]

    # Round scores for consistency
    analysis['overall_score'] = round(analysis['overall_score'], 1)
    for category in analysis['categories']:
        category['score'] = round(category['score'], 1)

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