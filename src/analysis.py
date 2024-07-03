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
        Analyze the following Terms of Service for {company_name} and provide a JSON output with the following structure:
        {{
            "categories": [
                {{
                    "name": "Clarity and Readability",
                    "weight": 20,
                    "score": 7,
                    "explanation": "Detailed explanation here"
                }},
                {{
                    "name": "Privacy and Data Security",
                    "weight": 20,
                    "score": 8,
                    "explanation": "Detailed explanation here"
                }},
                {{
                    "name": "User Rights and Ownership",
                    "weight": 15,
                    "score": 7,
                    "explanation": "Detailed explanation here"
                }},
                {{
                    "name": "Liability and Disclaimers",
                    "weight": 15,
                    "score": 6,
                    "explanation": "Detailed explanation here"
                }},
                {{
                    "name": "Termination and Account Suspension",
                    "weight": 10,
                    "score": 8,
                    "explanation": "Detailed explanation here"
                }},
                {{
                    "name": "Advertising and Third-Party Interactions",
                    "weight": 10,
                    "score": 7,
                    "explanation": "Detailed explanation here"
                }},
                {{
                    "name": "Dispute Resolution",
                    "weight": 5,
                    "score": 5,
                    "explanation": "Detailed explanation here"
                }},
                {{
                    "name": "Changes to Terms",
                    "weight": 5,
                    "score": 7,
                    "explanation": "Detailed explanation here"
                }}
            ],
            "overall_score": 6.95,
            "summary": "Overall summary of the analysis"
        }}

        Analyze the Terms of Service and fill in the JSON structure with appropriate scores, explanations, and summary.
        Ensure that your response is valid JSON and follows this exact structure.
        Provide detailed explanations for each category score.
        Calculate the overall_score as a weighted average of the category scores.
        Provide a comprehensive summary of the analysis.
        Do not include any text before or after the JSON object.

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

        # Parse the JSON response
        try:
            analysis_json = json.loads(response_text)
            
            # Validate and clean up the analysis result
            cleaned_analysis = {
                "categories": [],
                "overall_score": 0,
                "summary": ""
            }

            total_weight = 0
            weighted_score_sum = 0

            for category in analysis_json.get("categories", []):
                cleaned_category = {
                    "name": category.get("name", "Unnamed Category"),
                    "weight": int(category.get("weight", 0)),
                    "score": float(category.get("score", 0)),
                    "explanation": category.get("explanation", "No explanation provided.")
                }
                cleaned_analysis["categories"].append(cleaned_category)
                
                total_weight += cleaned_category["weight"]
                weighted_score_sum += cleaned_category["score"] * cleaned_category["weight"]

            if total_weight > 0:
                cleaned_analysis["overall_score"] = weighted_score_sum / total_weight
            else:
                cleaned_analysis["overall_score"] = 0

            cleaned_analysis["summary"] = analysis_json.get("summary", "No summary provided.")

            return cleaned_analysis

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse API response as JSON: {e}")
            return {
                "error": "Failed to parse the API response as JSON.",
                "raw_response": response_text[:1000]  # Include first 1000 characters of the raw response
            }

    except Exception as e:
        logger.exception("An error occurred while analyzing the Terms of Service")
        return {"error": f"An error occurred while analyzing the Terms of Service: {str(e)}"}

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