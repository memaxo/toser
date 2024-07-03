from flask import Flask, request, jsonify, render_template
from analysis import fetch_tos_document, extract_company_name, analyze_tos
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        logger.debug(f"Received data: {data}")

        url = data.get('url')
        logger.info(f"Received URL for analysis: {url}")

        if not url:
            logger.warning("No URL provided")
            return jsonify({'error': 'No URL provided'}), 400

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
        expected_keys = {"categories", "overall_score", "summary"}
        if not all(key in analysis for key in expected_keys):
            logger.error("Analysis result is missing expected keys")
            return jsonify({'error': 'Invalid analysis result structure'}), 500

        for category in analysis["categories"]:
            if not all(key in category for key in ["name", "weight", "score", "explanation"]):
                logger.error("Category is missing expected keys")
                return jsonify({'error': 'Invalid category structure in analysis result'}), 500

        logger.info("Analysis completed successfully")
        return jsonify(analysis)

    except Exception as e:
        logger.exception("An unexpected error occurred during analysis")
        return jsonify({'error': 'An unexpected error occurred', 'details': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.exception("An internal server error occurred")
    return jsonify({'error': 'Internal server error', 'details': str(error)}), 500

if __name__ == '__main__':
    app.run(debug=True)