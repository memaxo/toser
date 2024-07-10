from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from analysis import fetch_tos_document, extract_company_name, analyze_tos
import logging
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import google.generativeai as genai
import os
from models import db, User, Analysis
from datetime import datetime
import json

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get("SECRET_KEY", "your_secret_key")
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///asklivie.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

with app.app_context():
    db.create_all()

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

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        existing_user = User.query.filter((User.username == username) | (User.email == email)).first()
        if existing_user:
            flash('Username or email already exists')
            return redirect(url_for('register'))
        
        new_user = User(username=username, email=email)
        new_user.set_password(password)
        try:
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            flash('An error occurred. Please try again.')
            app.logger.error(f"Error during registration: {str(e)}")
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        if user is None:
            flash('No user found with that username')
        elif not user.check_password(password):
            flash('Incorrect password')
        else:
            login_user(user)
            user.last_login = datetime.utcnow()
            db.session.commit()
            return redirect(url_for('dashboard'))
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    analyses = Analysis.query.filter_by(user_id=current_user.id).all()
    return render_template('dashboard.html', analyses=analyses)

@app.route('/analyze', methods=['POST'])
@limiter.limit("5 per minute")
@login_required
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

        # Save the analysis to the database
        new_analysis = Analysis(
            url=url,
            company_name=company_name,
            result=json.dumps(analysis),  # Serialize the analysis dict to a JSON string
            user_id=current_user.id
        )
        db.session.add(new_analysis)
        db.session.commit()

        logger.info("Analysis completed successfully and saved to database")
        logger.debug(f"Sending analysis to frontend: {analysis}")
        return jsonify(analysis)

    except Exception as e:
        logger.exception("An unexpected error occurred during analysis")
        return jsonify({'error': 'An unexpected error occurred', 'details': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
