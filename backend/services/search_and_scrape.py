import os
import time
import json
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional, Any, Tuple
from urllib.parse import urlparse, urljoin
import logging
from PIL import Image
from io import BytesIO
import numpy as np
import traceback

# Assuming EmbeddingService is in the same directory or properly pathed
from .embed_and_search import EmbeddingService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration from environment
USER_AGENT = {'User-Agent': 'ShopSmarterBot/1.0'}
REQUEST_TIMEOUT = 10  # seconds
MAX_RETRIES = 3

# Site-specific selectors used as fallbacks
SITE_SELECTORS = {
    'amazon.': {
        'title': '#productTitle',
        'price': '#priceblock_ourprice, #priceblock_dealprice, #priceblock_saleprice',
        'image': '#landingImage'
    },
    'myntra.': {'title': 'h1.pdp-title', 'price': 'span.pdp-price', 'image': 'img.pdp-image'},
    'flipkart.': {'title': 'span.B_NuCI', 'price': 'div._30jeq3._16Jk6d', 'image': 'img._396cs4'},
    'tatacliq.': {'title': 'h1.pdp-title', 'price': 'div.price', 'image': 'img.primary-image'},
    'ajio.': {'title': 'div.product-title', 'price': 'div.price', 'image': 'img.product-image'}
}

# Helper function to safely get nested dictionary values
def _get_nested_value(data: Dict, keys: List[str], default: Any = None) -> Any:
    for key in keys:
        if isinstance(data, dict) and key in data:
            data = data[key]
        elif isinstance(data, list) and isinstance(key, int) and 0 <= key < len(data):
            data = data[key]
        else:
            return default
    return data

class SearchScrapeService:
    def __init__(self, embedding_service: Optional[EmbeddingService] = None):
        self.google_api_key = os.getenv('GOOGLE_API_KEY')
        self.google_cx = os.getenv('GOOGLE_CX')
        
        if not self.google_api_key or not self.google_cx:
            # Log a warning but don't raise an error immediately if web search might be optional
            # The methods using these should handle their absence.
            logger.warning('Google API key or CX missing. Web search functionality will be affected.')
            # raise EnvironmentError('Missing Google API credentials') # Or raise if it's critical

        # Selenium initialization - kept for now if it was intended as a fallback, but ensure it handles failure gracefully
        # If not used, this section can be removed along with its imports.
        try:
            # from selenium import webdriver # Moved imports to be conditional or ensure they are at top
            # from selenium.webdriver.chrome.options import Options
            # from selenium.webdriver.chrome.service import Service
            # from webdriver_manager.chrome import ChromeDriverManager
            # chrome_opts = Options()
            # chrome_opts.add_argument('--headless')
            # chrome_opts.add_argument('--disable-gpu')
            # chrome_opts.add_argument('--no-sandbox')
            # chrome_opts.add_argument('--disable-dev-shm-usage')
            # service = Service(ChromeDriverManager().install())
            # self.driver = webdriver.Chrome(service=service, options=chrome_opts)
            # logger.info('Selenium WebDriver initialized')
            self.driver = None # Placeholder: remove if Selenium fully removed
            logger.info('Selenium WebDriver initialization skipped/placeholder.')
        except Exception as e:
            logger.error(f'Could not init Selenium WebDriver: {e}')
            self.driver = None

        self.embedding_service = embedding_service or EmbeddingService()
        
        self.SITE_SELECTORS = {
            "amazon.in": {
                "title": ["span#productTitle"],
                "price": ["span.a-price-whole", "span.priceToPay span.a-offscreen"],
                "image": ["img#landingImage", "div.imgTagWrapper img"],
            },
            "flipkart.com": {
                "title": ["span.B_NuCI"],
                "price": ["div._30jeq3._16Jk6d"],
                "image": ["img._396cs4._2amPTt._3qGmMb", "img._2r_T1I"],
            },
            "myntra.com": {
                "title": ["h1.pdp-title"],
                "price": ["span.pdp-price strong", "span.pdp-mrp"],
                "image": ["div.image-grid-image img"],
            },
            "default": {
                "title": ["h1"],
                "price": [".price", ".Price", "#price", "#Price"],
                "image": ["img[alt*='product']", "img[src*='product']"],
            }
        }

    def _get_site_specific_selectors(self, domain: str) -> Dict[str, List[str]]:
        return self.SITE_SELECTORS.get(domain, self.SITE_SELECTORS["default"])

    def fetch_search_results(self, query: str, num_results: int = 20) -> List[str]:
        if not self.google_api_key or not self.google_cx:
            logger.error("Google API Key or CX not configured for SearchScrapeService instance.")
            return []
        
        search_endpoint = 'https://www.googleapis.com/customsearch/v1'
        params = {
            "key": self.google_api_key,
            "cx": self.google_cx,
            "q": query,
            "num": num_results,
        }
        try:
            response = requests.get(search_endpoint, params=params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status() # This will raise an HTTPError for 4xx/5xx responses
            results = response.json()
            return [item["link"] for item in results.get("items", []) if "link" in item]
        except requests.exceptions.HTTPError as e:
            # Attempt to get more details from Google's JSON error response
            error_details = "No additional error details from Google."
            try:
                google_error_json = e.response.json()
                if 'error' in google_error_json and 'message' in google_error_json['error']:
                    error_details = f"Google API Error Message: {google_error_json['error']['message']}"
                elif 'error' in google_error_json:
                    error_details = f"Google API Error (raw): {google_error_json['error']}"
                else:
                    error_details = f"Google API Response (non-JSON or unexpected format): {e.response.text[:500]}" # Log first 500 chars
            except json.JSONDecodeError:
                error_details = f"Google API Response (not JSON): {e.response.text[:500]}" # Log first 500 chars if not JSON
            except Exception as inner_e:
                error_details = f"Failed to parse Google error response: {inner_e}"
            
            logger.error(f"Google API request failed: {e}. {error_details}")
            return []
        except requests.exceptions.RequestException as e: # Other network issues (timeout, DNS, etc.)
            logger.error(f"Google API request failed (network/other): {e}")
            return []
        except Exception as e: # Catch-all for any other unexpected errors
            logger.error(f"Error processing Google search results: {e}")
            return []

    def extract_favicon(self, url: str) -> str:
        try:
            domain = urlparse(url).netloc
            favicon = f"https://{domain}/favicon.ico"
            logger.debug(f'Extracted favicon URL: {favicon}')
            return favicon
        except Exception as e:
            logger.error(f'Error extracting favicon for {url}: {e}')
            return ""

    def scrape_product(self, url: str) -> Optional[Dict[str, Any]]:
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")

            title: Optional[str] = None
            price: Optional[float] = None
            price_currency: str = "INR" # Default currency
            image_url: Optional[str] = None
            
            # 1. Attempt to parse JSON-LD
            ld_script = soup.find('script', type='application/ld+json')
            if ld_script and ld_script.string:
                try:
                    ld_data_list = json.loads(ld_script.string)
                    # Ensure ld_data is a dictionary representing the main product
                    if isinstance(ld_data_list, list):
                        product_ld = next((item for item in ld_data_list if isinstance(item, dict) and item.get("@type") == "Product"), None)
                        if not product_ld and ld_data_list: # Fallback to the first item if no explicit Product type found
                            product_ld = ld_data_list[0] if isinstance(ld_data_list[0], dict) else {}
                        ld_data = product_ld if product_ld else {}
                    elif isinstance(ld_data_list, dict):
                        ld_data = ld_data_list # It's already a dictionary
                    else:
                        ld_data = {}

                    if isinstance(ld_data, dict) and ld_data: # Proceed if we have a valid product dictionary
                        logger.info(f"Found JSON-LD data for {url}")
                        title = _get_nested_value(ld_data, ['name'], title)
                        
                        # Enhanced offer parsing to prioritize INR
                        offers_data_raw = _get_nested_value(ld_data, ['offers'])
                        selected_offer = None

                        if isinstance(offers_data_raw, list): # Multiple offers
                            # Try to find an offer with INR currency
                            for offer_item in offers_data_raw:
                                if isinstance(offer_item, dict) and _get_nested_value(offer_item, ['priceCurrency'], "").upper() == "INR":
                                    selected_offer = offer_item
                                    logger.debug(f"Found INR offer in JSON-LD list for {url}")
                                    break
                            if not selected_offer and offers_data_raw: # If no INR offer, take the first one
                                selected_offer = offers_data_raw[0] if isinstance(offers_data_raw[0], dict) else {}
                                logger.debug(f"No INR offer in JSON-LD list, taking first available for {url}")
                        elif isinstance(offers_data_raw, dict): # Single offer
                            selected_offer = offers_data_raw
                            logger.debug(f"Found single offer in JSON-LD for {url}")
                        
                        if selected_offer:
                            current_offer_currency = _get_nested_value(selected_offer, ['priceCurrency'], "").upper()
                            price_str_raw = str(_get_nested_value(selected_offer, ['price'], "")) or \
                                         str(_get_nested_value(selected_offer, ['lowPrice'], "")) or \
                                         str(_get_nested_value(selected_offer, ['highPrice'], ""))
                            
                            if price_str_raw:
                                try:
                                    # More robust price cleaning
                                    cleaned_price_str = ''.join(filter(lambda x: x.isdigit() or x == '.', price_str_raw))
                                    if cleaned_price_str:
                                        parsed_price = float(cleaned_price_str)
                                        # Only update if we got a valid price and currency is INR, or if no price is set yet
                                        if current_offer_currency == "INR":
                                            price = parsed_price
                                            price_currency = "INR"
                                            logger.info(f"Price from JSON-LD (INR): {price} {price_currency} for {url}")
                                        elif not price and current_offer_currency: # If no INR price found yet, take what's available
                                            price = parsed_price
                                            price_currency = current_offer_currency
                                            logger.info(f"Price from JSON-LD (non-INR, fallback): {price} {price_currency} for {url}")
                                        elif not price: # No currency info, but got a price
                                            price = parsed_price
                                            logger.info(f"Price from JSON-LD (currency unknown): {price} for {url}")


                                except ValueError:
                                    logger.warning(f"Could not parse price '{price_str_raw}' from JSON-LD for {url}")
                        
                        img_obj = _get_nested_value(ld_data, ['image'])
                        if isinstance(img_obj, list):
                            image_url = _get_nested_value(img_obj, [0, 'url'], _get_nested_value(img_obj, [0], image_url))
                        elif isinstance(img_obj, dict):
                            image_url = _get_nested_value(img_obj, ['url'], image_url)
                        elif isinstance(img_obj, str):
                            image_url = img_obj
                        logger.debug(f"JSON-LD extracted: Title='{title}', Price='{price}', Currency='{price_currency}', Image='{image_url}'")
                except json.JSONDecodeError:
                    logger.warning(f"Could not decode JSON-LD for {url}")
                except Exception as e:
                    logger.error(f"Error parsing JSON-LD for {url}: {e}. Data: {str(ld_script.string)[:200]}")

            domain = urlparse(url).netloc.replace("www.", "")
            selectors = self._get_site_specific_selectors(domain)

            # Fallback to CSS selectors if JSON-LD didn't provide all info or failed
            if not title:
                for selector in selectors.get("title", []):
                    element = soup.select_one(selector)
                    if element: title = element.get_text(strip=True); break
            
            if price is None: # Only use CSS selector for price if JSON-LD didn't yield one
                logger.debug(f"Price not found via JSON-LD for {url}. Trying CSS selectors.")
                for selector in selectors.get("price", []):
                    element = soup.select_one(selector)
                    if element:
                        price_text = element.get_text(strip=True)
                        # Try to infer currency or assume INR if specific symbols present
                        if '₹' in price_text or 'INR' in price_text.upper():
                            cleaned_price_str = ''.join(filter(lambda x: x.isdigit() or x == '.', price_text))
                            try: 
                                price = float(cleaned_price_str)
                                price_currency = "INR" # Explicitly set/confirm INR
                                logger.info(f"Price from CSS (INR detected): {price} {price_currency} for {url}")
                                break
                            except ValueError: 
                                logger.warning(f"Could not parse price '{price_text}' from CSS selector for {url}")
                                continue
                        elif '$' in price_text or 'USD' in price_text.upper():
                             # If we find a USD price and have no INR price, we might take it but log it.
                            cleaned_price_str = ''.join(filter(lambda x: x.isdigit() or x == '.', price_text))
                            try:
                                potential_usd_price = float(cleaned_price_str)
                                if price is None: # Only take if no price has been set at all
                                    price = potential_usd_price
                                    price_currency = "USD"
                                    logger.info(f"Price from CSS (USD detected, fallback): {price} {price_currency} for {url}")
                                    break 
                                # else: we already have an INR price or a preferred JSON-LD price, so ignore this USD one.
                            except ValueError:
                                logger.warning(f"Could not parse USD price '{price_text}' from CSS selector for {url}")
                                continue
                        else: # Generic price, assume INR as default context
                            cleaned_price_str = ''.join(filter(lambda x: x.isdigit() or x == '.', price_text))
                            try:
                                price = float(cleaned_price_str)
                                price_currency = "INR" # Assume INR if no other indicators
                                logger.info(f"Price from CSS (assumed INR): {price} {price_currency} for {url}")
                                break
                            except ValueError: 
                                logger.warning(f"Could not parse generic price '{price_text}' from CSS selector for {url}")
                                continue
            
            if not image_url:
                for selector in selectors.get("image", []):
                    element = soup.select_one(selector)
                    if element and element.get("src"): image_url = urljoin(url, element["src"]); break
            
            if title and price is not None and image_url:
                # Ensure price is float, default to 0.0 if somehow None (should be caught by 'price is not None')
                final_price = float(price) if price is not None else 0.0 
                logger.info(f"Scraped: Title='{title}', Price='{final_price}', Currency='{price_currency}', Image='{image_url}' from {url}")
                return {"title": title, "price": final_price, "price_currency": price_currency, "image_url": image_url, "source": domain, "link": url}
            else:
                logger.warning(f"Incomplete scrape for {url}. T:{title}, P:{price} ({price_currency}), I:{image_url}")
                return None
        except requests.exceptions.RequestException as e: logger.error(f"Fetch failed for {url}: {e}"); return None
        except Exception as e: 
            logger.error(f"General scraping error for {url}: {e}")
            logger.error(traceback.format_exc()) # Log full traceback for unexpected errors
            return None

    def search_and_scrape(self, query: str, query_image: Optional[Image.Image] = None, top_n_search: int = 10, top_n_rerank: int = 10) -> List[Dict]:
        logger.info(f"Search: '{query}', top_n_search:{top_n_search}, top_n_rerank:{top_n_rerank}")
        product_urls = self.fetch_search_results(query, num_results=top_n_search)
        if not product_urls: logger.warning("No URLs from Google search."); return []

        scraped_products: List[Dict] = []
        product_images: List[Optional[Image.Image]] = []

        for url in product_urls:
            product_data = self.scrape_product(url)
            if product_data and product_data.get("image_url"):
                scraped_products.append(product_data)
                try:
                    response = requests.get(product_data["image_url"], timeout=10, stream=True)
                    response.raise_for_status()
                    product_images.append(Image.open(BytesIO(response.content)).convert("RGB"))
                except Exception as e:
                    logger.warning(f"Img download/open failed: {product_data['image_url']} ({e})")
                    product_images.append(None)
            if len(scraped_products) >= top_n_search: break
        
        if not scraped_products: logger.warning("No products scraped."); return []

        if query_image and self.embedding_service:
            logger.info("Visual reranking...")
            query_embedding = self.embedding_service.get_image_embedding(query_image)
            if query_embedding is None: logger.warning("Query image embedding failed. No rerank."); return scraped_products[:top_n_rerank]

            valid_products_for_rerank: List[Dict] = []
            product_embeddings_list: List[np.ndarray] = []
            for i, product_img in enumerate(product_images):
                if product_img:
                    scraped_product_embedding = self.embedding_service.get_image_embedding(product_img)
                    if scraped_product_embedding is not None:
                        valid_products_for_rerank.append(scraped_products[i])
                        product_embeddings_list.append(scraped_product_embedding)
            
            if not product_embeddings_list: logger.warning("No product embeddings for rerank. No rerank."); return scraped_products[:top_n_rerank]

            product_embeddings_np = np.concatenate(product_embeddings_list, axis=0)
            similarities = (query_embedding @ product_embeddings_np.T).flatten()
            for i, product in enumerate(valid_products_for_rerank): product["similarity"] = float(similarities[i])
            reranked_products = sorted(valid_products_for_rerank, key=lambda x: x["similarity"], reverse=True)
            logger.info(f"Reranked {len(reranked_products)} products. Top sim: {reranked_products[0]['similarity'] if reranked_products else 'N/A'}")
            return reranked_products[:top_n_rerank]
        else:
            logger.info("No query image or embedding service for rerank. Returning top N scraped.")
            return scraped_products[:top_n_rerank]

    def close(self):
        if self.driver:
            self.driver.quit()
            logger.info('Selenium WebDriver closed')

if __name__ == '__main__':
    svc = SearchScrapeService()
    try:
        results = svc.search_and_scrape('red leather jacket')
        print(json.dumps(results, indent=2))
    finally:
        svc.close()
