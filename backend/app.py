"""
ShopSmarter Flask Backend
Main API server that integrates search, embedding, and refinement services.
"""
import os
import json
import logging
import traceback
from typing import List, Dict, Optional
import time

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
from dotenv import load_dotenv

# For BLIP captioning (ensure transformers is installed: pip install transformers)
from transformers import BlipProcessor, BlipForConditionalGeneration

# Get the backend directory path
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))

# Load environment variables before anything else
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for more verbose logging
    format="%(asctime)s %(levelname)s %(name)s – %(message)s"
)
logger = logging.getLogger(__name__)
logger.info('Environment variables loaded')

# Initialize Flask app
app = Flask(__name__)
# Enable CORS for all routes with more permissive settings
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Accept"],
        "supports_credentials": True
    }
})
logger.info('Flask app initialized with CORS')

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Accept')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

# Import services with correct names
from services.search_and_scrape import SearchScrapeService
from services.embed_and_search import EmbeddingService
from services.refine_with_llm import RefinementService

# Initialize services and models
search_service: Optional[SearchScrapeService] = None
embedding_service: Optional[EmbeddingService] = None
refinement_service: Optional[RefinementService] = None
blip_processor: Optional[BlipProcessor] = None
blip_model: Optional[BlipForConditionalGeneration] = None

BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-large"

try:
    # Set environment variable for tags.json path
    os.environ['TAGS_JSON'] = os.path.join(BACKEND_DIR, 'tags.json')
    logger.debug(f"TAGS_JSON path set to: {os.environ['TAGS_JSON']}")
    
    embedding_service = EmbeddingService()
    logger.debug("EmbeddingService initialized")
    
    search_service = SearchScrapeService(embedding_service=embedding_service)
    logger.debug("SearchScrapeService initialized")
    
    refinement_service = RefinementService()
    logger.debug("RefinementService initialized")

    # Load BLIP model for captioning
    logger.info(f"Loading BLIP model: {BLIP_MODEL_NAME}")
    blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL_NAME)
    blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_NAME)
    logger.info("BLIP model loaded successfully.")
    
    logger.info('All services and models initialized successfully')
except ImportError as e:
    logger.error(f'Failed to import required modules: {str(e)}')
    logger.error(traceback.format_exc())
    raise
except Exception as e:
    logger.error(f'Failed to initialize services or models: {str(e)}')
    logger.error(traceback.format_exc())
    raise

# Configure upload settings
UPLOAD_FOLDER = os.path.abspath(os.getenv('UPLOAD_FOLDER', os.path.join(BACKEND_DIR, 'uploads')))
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
logger.debug(f"Upload folder set to: {UPLOAD_FOLDER}")

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    logger.info(f"Created upload folder: {UPLOAD_FOLDER}")

def allowed_file(filename: str) -> bool:
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/query', methods=['POST'])
def query_route():
    request_start_time = time.time()
    try:
        logger.info("Received API query request")
        if 'image' not in request.files:
            logger.error("No image file in request")
            return jsonify({'error': 'No image file provided'}), 400

        file = request.files['image']
        prompt = request.form.get('prompt', '')

        if file.filename == '':
            logger.error("Empty filename")
            return jsonify({'error': 'No selected file'}), 400
        if not allowed_file(file.filename):
            logger.error(f"Invalid file type: {file.filename}")
            return jsonify({'error': 'Invalid file type'}), 400

        time_before_save = time.time()
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        logger.info(f"File saved to: {filepath} in {time.time() - time_before_save:.2f}s")
        
        time_before_img_open = time.time()
        img_pil = Image.open(filepath).convert("RGB")
        logger.info(f"Image opened and converted in {time.time() - time_before_img_open:.2f}s")

        search_query_for_google: str
        query_generation_start_time = time.time()

        if prompt:
            logger.info(f"User provided prompt: '{prompt}'. Using it to generate Google search query.")
            search_query_for_google = refinement_service.generate_shopping_query(prompt)
            logger.info(f"LLM generated Google search query from prompt: {search_query_for_google} in {time.time() - query_generation_start_time:.2f}s")
        elif blip_model and blip_processor:
            logger.info("No user prompt. Generating image caption with BLIP.")
            blip_start_time = time.time()
            inputs = blip_processor(images=img_pil, return_tensors="pt")
            out = blip_model.generate(**inputs, max_new_tokens=50)
            caption = blip_processor.decode(out[0], skip_special_tokens=True)
            logger.info(f"BLIP caption: '{caption}' generated in {time.time() - blip_start_time:.2f}s")
            
            refine_blip_start_time = time.time()
            search_query_for_google = refinement_service.generate_shopping_query(caption)
            logger.info(f"LLM generated Google search query from BLIP caption in {time.time() - refine_blip_start_time:.2f}s. Total query gen: {time.time() - query_generation_start_time:.2f}s")
        else:
            logger.warning("BLIP model not available and no prompt. Falling back to tag-based search.")
            tags = embedding_service.get_image_tags(img_pil, top_k=5)
            sites = "site:amazon.in OR site:flipkart.com OR site:myntra.com"
            search_query_for_google = f"{' '.join(tags)} {sites}"
            logger.info(f"Fallback tag-based Google search query: {search_query_for_google} in {time.time() - query_generation_start_time:.2f}s")

        if not any(s in search_query_for_google for s in ["site:amazon.in", "site:flipkart.com", "site:myntra.com"]):
            search_query_for_google += " site:amazon.in OR site:flipkart.com OR site:myntra.com"
            logger.info(f"Appended site restrictions. Final Google query: {search_query_for_google}")

        search_scrape_start_time = time.time()
        products = search_service.search_and_scrape(
            query=search_query_for_google, 
            query_image=img_pil,
            top_n_search=10, 
            top_n_rerank=10
        )
        logger.info(f"Search and scrape returned {len(products)} products in {time.time() - search_scrape_start_time:.2f}s")

        total_request_time = time.time() - request_start_time
        logger.info(f"Total request processing time: {total_request_time:.2f}s")
        return jsonify({'products': products})

    except Exception as e:
        total_request_time = time.time() - request_start_time
        logger.error(f"Error processing API query: {str(e)} after {total_request_time:.2f}s of processing.")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Failed to process API query',
            'details': str(e)
        }), 500

if __name__ == '__main__':
    # Disable reloader to avoid termios.error in some environments
    app.run(debug=True, port=5001, use_reloader=False)
