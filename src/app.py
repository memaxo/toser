from flask import Flask, request, jsonify, render_template
from analysis import fetch_tos_document, extract_company_name, analyze_tos
import logging
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import google.generativeai as genai
import os

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configure rate limiting
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)
limiter.init_app(app)

# Configure Gemini API
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
@limiter.limit("5 per minute")
def analyze():
    try:
        data = request.json
        logger.debug(f"Received data: {data}")

        url = data.get('url')
        logger.info(f"Received URL for analysis: {url}")

        if not url:
            logger.warning("No URL provided")
            return jsonify({'error': 'No URL provided'}), 400

        if not url.startswith(('http://', 'https://')):
            logger.warning("Invalid URL format")
            return jsonify({'error': 'Invalid URL format. Please include http:// or https://'}), 400

        logger.info(f"Attempting to fetch ToS from URL: {url}")
        tos_text = fetch_tos_document(url)

        if not tos_text:
            logger.error(f"Unable to fetch the Terms of Service document from: {url}")
            return jsonify({'error': 'Unable to fetch the Terms of Service document. Please check the URL and try again.'}), 400

        company_name = extract_company_name(url)
        logger.info(f"Analyzing ToS for company: {company_name}")

        analysis = analyze_tos(tos_text, company_name)

        if "error" in analysis:
            logger.error(f"Error in analysis: {analysis['error']}")
            return jsonify(analysis), 400

        # Validate the structure of the analysis result
        required_keys = {"initial_assessment", "categories", "final_score", "letter_grade", "summary", "green_flags", "red_flags", "company_name"}
        if not all(key in analysis for key in required_keys):
            missing_keys = required_keys - set(analysis.keys())
            logger.error(f"Analysis result is missing expected keys: {missing_keys}")
            return jsonify({'error': f'Invalid analysis result structure. Missing keys: {missing_keys}'}), 500

        for category in analysis.get("categories", []):
            if not all(key in category for key in ["name", "user_friendly_aspect", "concerning_aspect", "score", "justification"]):
                logger.error("Category is missing expected keys")
                return jsonify({'error': 'Invalid category structure in analysis result'}), 500

        # Ensure final_score is a float
        try:
            analysis['final_score'] = float(analysis['final_score'])
        except ValueError:
            logger.error("Invalid final_score value")
            analysis['final_score'] = 0.0

        # Ensure green_flags and red_flags are lists
        analysis['green_flags'] = list(analysis.get('green_flags', []))
        analysis['red_flags'] = list(analysis.get('red_flags', []))

        # Ensure summary is a string
        analysis['summary'] = str(analysis.get('summary', 'No summary available.'))

        logger.info("Analysis completed successfully")
        logger.debug(f"Sending analysis to frontend: {analysis}")
        return jsonify(analysis)

    except Exception as e:
        logger.exception("An unexpected error occurred during analysis")
        return jsonify({'error': 'An unexpected error occurred', 'details': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)