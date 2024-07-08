# ToSer - Terms of Service Analyzer

ToSer is an AI-powered web application that analyzes Terms of Service documents, providing users with easy-to-understand summaries and scores across various categories.

## Features

- Analyze Terms of Service documents from any URL
- AI-powered assessment using Google's Gemini API
- Breakdown of analysis across multiple categories:
  - Clarity and Readability
  - Privacy and Data Security
  - User Rights and Ownership
  - Liability and Disclaimers
  - Termination and Account Suspension
  - Advertising and Third-Party Interactions
  - Dispute Resolution
  - Changes to Terms
- Overall score and detailed summary
- Visual representation of category scores
- Responsive web interface

## Prerequisites

- Python 3.7+
- Flask
- BeautifulSoup4
- Requests
- tldextract
- google-generativeai library
- A valid Google API key for the Gemini API

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/toser.git
   cd toser
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up your Google API key:
   - Obtain a Gemini API key from the Google AI Studio
   - Set it as an environment variable:
     ```
     export GEMINI_API_KEY='your-api-key-here'
     ```
     On Windows, use `set GEMINI_API_KEY=your-api-key-here`

5. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

6. Start the Flask application:
   ```
   python app.py
   ```

7. Open a web browser and navigate to `http://localhost:5000`

## Running Tests

To run the tests, follow these steps:

1. Ensure you have installed all the required dependencies as mentioned in the Setup section.

2. Run the tests using the following command from the project root directory:
   ```
   PYTHONPATH=./src python -m pytest tests/
   ```
   
   On Windows, use:
   ```
   set PYTHONPATH=./src && python -m pytest tests/
   ```

## Usage

1. Enter the URL of a Terms of Service document in the input field.
2. Click the "Analyze" button.
3. Wait for the analysis to complete.
4. Review the results, including:
   - Overall score
   - Category breakdown with individual scores and explanations
   - Visual chart of category scores
   - Summary of the analysis

## Project Structure

- `app.py`: Main Flask application
- `analysis.py`: Contains the logic for fetching and analyzing ToS documents
- `templates/index.html`: Frontend HTML template
- `static/`: Directory for static assets (CSS, JS)

## Contributing

Contributions to ToSer are welcome! Please feel free to submit pull requests, create issues or spread the word.

## License

This project is proprietary software. All rights reserved.

Copyright (c) 2024 Jack Mazac

Unauthorized copying, use, distribution, or modification of this software, or any portion of it, is strictly prohibited. This software is provided "AS IS," without warranty of any kind, express or implied.

For inquiries about licensing or usage, please contact jack@aibridgelabs.com.

## Acknowledgments

- Google Generative AI for providing the Gemini API
- All contributors and users of ToSer

## Disclaimer

ToSer is a tool to assist in understanding Terms of Service documents. It should not be considered as legal advice. Always consult with a legal professional for legal matters.
