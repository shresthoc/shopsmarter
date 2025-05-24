"""
Refinement Service for ShopSmarter
Analyzes user queries and current results to extract filter criteria via LLM,
applies those filters, then re-ranks using a second LLM call for accuracy.
"""
import os
import logging
import json
from typing import List, Dict, Optional, Tuple
import uuid
import re
import traceback

import spacy
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from transformers import pipeline as hf_pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)

# Model configuration (CPU only)
MODEL_NAME = os.getenv("REFINE_MODEL", "facebook/opt-350m")

class FilterCriteria(BaseModel):
    """Structured filter criteria for product refinement."""
    min_price: Optional[float] = Field(None, description="Minimum price to include")
    max_price: Optional[float] = Field(None, description="Maximum price to include")
    style_keywords: List[str] = Field(default_factory=list, description="List of style descriptors to include")
    color_preferences: List[str] = Field(default_factory=list, description="List of preferred colors")
    excluded_terms: List[str] = Field(default_factory=list, description="List of terms to exclude from titles")
    sort_by: str = Field("relevance", description="Sorting criteria: relevance, price_low, price_high")

class RefinementService:
    def __init__(self):
        # Load spaCy model for text analysis
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded spaCy model successfully")
        except OSError:
            logger.error("spaCy model en_core_web_sm not found. Run: python -m spacy download en_core_web_sm")
            raise
        except Exception as e:
            logger.error(f"Failed to load spaCy model: {str(e)}")
            raise

        # Initialize HF text-generation pipelines on CPU
        # MODEL_NAME is fetched from os.getenv("REFINE_MODEL", ...) inside the try block
        model_name_for_llm = os.getenv("REFINE_MODEL", "facebook/opt-350m") # Default if not set
        logger.info(f"Attempting to load REFINE_MODEL: {model_name_for_llm}")

        try:
            # General purpose generator for tasks like filter extraction, reranking
            self.general_generator = hf_pipeline(
                task="text-generation",
                model=model_name_for_llm, # Use the fetched model name
                max_new_tokens=150,
                device=-1,  # CPU only
                trust_remote_code=True # Added for models like Phi
            )
            self.llm = HuggingFacePipeline(pipeline=self.general_generator)
            logger.info(f"Initialized general HuggingFace pipeline with model {model_name_for_llm}.")

            # Specialized generator for concise Google query generation
            self.query_gen_generator = hf_pipeline(
                task="text-generation",
                model=model_name_for_llm, # Use the fetched model name
                max_new_tokens=40,
                device=-1,  # CPU only
                trust_remote_code=True # Added for models like Phi
            )
            self.query_gen_llm = HuggingFacePipeline(pipeline=self.query_gen_generator)
            logger.info(f"Initialized query-focused HuggingFace pipeline with model {model_name_for_llm}.")

        except Exception as e:
            logger.error(f"Failed to initialize HuggingFace pipelines with model {model_name_for_llm}: {str(e)}")
            logger.error(traceback.format_exc()) # Log full traceback
            raise

        # Prompt template for query analysis
        self.parser = PydanticOutputParser(pydantic_object=FilterCriteria)
        self.query_analysis_prompt = PromptTemplate(
            input_variables=["query", "current_results", "format_instructions"],
            template=(
                "You are a shopping assistant.\n"
                "Extract filter criteria from the user query and sample results.\n"
                "Return JSON with keys: min_price, max_price, style_keywords, color_preferences, excluded_terms, sort_by.\n"
                "User Query: {query}\n"
                "Sample Results:\n{current_results}\n"
                "{format_instructions}"
            )
        )
        self.filter_extraction_chain = LLMChain(
            llm=self.llm, # Uses general llm
            prompt=self.query_analysis_prompt,
            output_parser=self.parser
        )

        # Prompt template for re-ranking
        self.product_reranking_prompt = PromptTemplate(
            input_variables=["products", "criteria"],
            template=(
                "Given the filtered products and criteria, rank them by relevance.\n"
                "Products (id and title list): {products}\n"
                "Criteria: {criteria}\n"
                "Return a JSON list of product IDs in ranked order."
            )
        )
        self.product_reranking_chain = LLMChain(
            llm=self.llm, # Uses general llm
            prompt=self.product_reranking_prompt
        )

        # Prompt template for Google query generation
        self.google_query_generation_prompt = PromptTemplate(
            input_variables=["text_input"],
            template=(
                "You are an expert at creating concise Google search queries for e-commerce product discovery. "
                "Convert the following input text into a short, effective Google search query. "
                "The query should focus on key visual attributes, product type, and essential keywords. "
                "It must be suitable for a search engine. Output only the query string itself, without any preamble or extra examples.\n\n"
                "Example 1:\n"
                "Input Text: A beautiful long flowing blue summer dress with floral patterns, perfect for weddings and outdoor parties. It is made of cotton.\n"
                "Google Search Query: blue floral cotton summer dress wedding party\n\n"
                "Example 2:\n"
                "Input Text: High-tech noise cancelling headphones, black, wireless, for travel and office use. Excellent battery life.\n"
                "Google Search Query: black wireless noise cancelling travel office headphones\n\n"
                "Example 3:\n"
                "Input Text: a vibrant red t-shirt for men, pure cotton, with a small white embroidered logo on the chest, crew neck, suitable for casual wear.\n"
                "Google Search Query: mens red cotton t-shirt white logo crew neck casual\n\n"
                "Input Text: \"{text_input}\"\n"
                "Google Search Query:"
            )
        )
        self.google_query_generation_chain = LLMChain(
            llm=self.query_gen_llm, # Uses query_gen_llm with shorter max_new_tokens
            prompt=self.google_query_generation_prompt
        )

        logger.info("RefinementService initialized with LLM chains (including few-shot for query gen) and spaCy.")

    def extract_price_range(self, text: str) -> Tuple[Optional[float], Optional[float]]:
        """Extract price range from text using spaCy."""
        if not text:
            return None, None

        try:
            doc = self.nlp(text.lower())
            prices = []
            for token in doc:
                if token.like_num:
                    prev = doc[token.i-1].text if token.i > 0 else ""
                    if prev in ['$', '£', '€', '₹']:
                        try:
                            prices.append(float(token.text))
                        except ValueError:
                            continue
            if len(prices) >= 2:
                return min(prices), max(prices)
            if len(prices) == 1:
                txt = text.lower()
                if any(w in txt for w in ['under', 'less than', 'cheaper than']):
                    return None, prices[0]
                if any(w in txt for w in ['over', 'more than', 'above']):
                    return prices[0], None
        except Exception as e:
            logger.error(f"Error extracting price range: {str(e)}")
        return None, None

    def extract_style_keywords(self, text: str) -> List[str]:
        """Extract style keywords from text using spaCy."""
        if not text:
            return []

        try:
            doc = self.nlp(text.lower())
            styles = set()
            for token in doc:
                if token.pos_ in ['ADJ', 'NOUN']:
                    lemma = token.lemma_
                    if lemma in ['casual','formal','elegant','modern','vintage','classic','trendy','bohemian','minimalist','luxury']:
                        styles.add(lemma)
            return list(styles)
        except Exception as e:
            logger.error(f"Error extracting style keywords: {str(e)}")
            return []

    def analyze_query(self, query: str, current_results: List[Dict]) -> FilterCriteria:
        """Analyze user query and current results to extract filter criteria."""
        if not query or not current_results:
            return FilterCriteria()

        try:
            # Add unique IDs to products if not present
            for p in current_results:
                if 'id' not in p:
                    p['id'] = str(uuid.uuid4())

            summary = "\n".join(f"- {p['title']} ({p.get('price','N/A')})" for p in current_results[:3])
            instructions = self.parser.get_format_instructions()
            logger.debug("Analyzing query with LLMChain")
            raw = self.filter_extraction_chain.run({
                'query': query,
                'current_results': summary,
                'format_instructions': instructions
            })
            try:
                criteria = FilterCriteria.parse_obj(raw)
                logger.info(f"Parsed criteria: {criteria}")
                return criteria
            except Exception as e:
                logger.error(f"Failed to parse criteria, falling back: {str(e)}")
        except Exception as e:
            logger.error(f"Error in analyze_query: {str(e)}")

        # Fallback to basic extraction
        min_p, max_p = self.extract_price_range(query)
        styles = self.extract_style_keywords(query)
        return FilterCriteria(
            min_price=min_p,
            max_price=max_p,
            style_keywords=styles,
            color_preferences=[],
            excluded_terms=[],
            sort_by="relevance"
        )

    def apply_filters(self, products: List[Dict], criteria: FilterCriteria) -> List[Dict]:
        """Apply filter criteria to product list."""
        if not products:
            return []

        try:
            filtered = products[:]
            # Price filters
            if criteria.min_price is not None:
                filtered = [p for p in filtered if p.get('price') is not None and p['price'] >= criteria.min_price]
            if criteria.max_price is not None:
                filtered = [p for p in filtered if p.get('price') is not None and p['price'] <= criteria.max_price]
            # Style/color filters
            if criteria.style_keywords or criteria.color_preferences:
                keys = [k.lower() for k in criteria.style_keywords + criteria.color_preferences]
                filtered = [p for p in filtered if any(k in p['title'].lower() for k in keys)]
            # Exclusions
            if criteria.excluded_terms:
                exclude = [t.lower() for t in criteria.excluded_terms]
                filtered = [p for p in filtered if not any(t in p['title'].lower() for t in exclude)]
            return filtered
        except Exception as e:
            logger.error(f"Error applying filters: {str(e)}")
            return products

    def rerank(self, filtered: List[Dict], criteria: FilterCriteria) -> List[Dict]:
        """Re-rank filtered products based on criteria."""
        if not filtered:
            return []

        try:
            # Ensure all products have IDs
            for p in filtered:
                if 'id' not in p:
                    p['id'] = str(uuid.uuid4())

            # Prepare product id-title pairs
            products_list = json.dumps([{ 'id': p['id'], 'title': p['title'] } for p in filtered])
            criteria_json = criteria.json()
            logger.debug("Re-ranking filtered products via LLMChain")
            raw = self.product_reranking_chain.run({ 'products': products_list, 'criteria': criteria_json })
            try:
                ranked_ids = json.loads(raw)
                id_to_prod = {p['id']: p for p in filtered}
                return [id_to_prod[i] for i in ranked_ids if i in id_to_prod]
            except Exception as e:
                logger.error(f"Failed to parse ranking output: {str(e)}")
        except Exception as e:
            logger.error(f"Error in rerank: {str(e)}")

        # Fallback to basic sorting
        if criteria.sort_by == "price_low":
            return sorted(filtered, key=lambda x: x.get('price', float('inf')))
        elif criteria.sort_by == "price_high":
            return sorted(filtered, key=lambda x: x.get('price', 0), reverse=True)
        return filtered

    def generate_shopping_query(self, text_input: str, max_length: int = 25) -> str:
        """
        Uses an LLM to convert a descriptive text (user prompt or image caption)
        into a concise Google shopping query.
        """
        if not text_input:
            logger.warning("generate_shopping_query called with empty input.")
            return ""

        generated_query = ""
        try:
            logger.info(f"Generating Google shopping query from text: '{text_input[:100]}...' using LLM.")
            raw_llm_output = self.google_query_generation_chain.run(text_input=text_input)
            logger.debug(f"Raw LLM output for query generation: {raw_llm_output!r}")

            lines = [line.strip() for line in raw_llm_output.splitlines()]
            if lines and lines[0]:
                generated_query = lines[0]
            else:
                generated_query = raw_llm_output

            generated_query = generated_query.strip('\"\' \n\t')

            if "Google Search Query:" in generated_query:
                generated_query = generated_query.split("Google Search Query:")[-1].strip('\"\' \n\t')

            if not generated_query:
                logger.warning(f"LLM returned an empty query after initial parsing from: '{text_input}'. Will proceed to length check / fallback.")

            if not generated_query or len(generated_query.split()) > 15:
                warning_msg = f"LLM generated unusable query (empty or too long: '{len(generated_query.split())}' words from '{generated_query[:100]}...')."
                logger.warning(warning_msg)
                raise ValueError(warning_msg)

        except Exception as e:
            logger.error(f"LLM (google_query_generation_chain) failed or produced unusable output for input '{text_input}': {e}")
            logger.warning("Falling back to basic spaCy keyword extraction for Google query.")
            try:
                doc = self.nlp(text_input)
                generic_terms_to_remove = {"picture", "image", "photo", "graphic", "drawing", "art", "photo of", "image of", "a picture of"}
                keywords = [
                    token.lemma_.lower()
                    for token in doc
                    if token.pos_ in ["NOUN", "PROPN", "ADJ"]
                    and not token.is_stop
                    and len(token.lemma_) > 2
                    and token.lemma_.lower() not in generic_terms_to_remove
                ]
                generated_query = " ".join(list(dict.fromkeys(keywords))[:7])
                if not generated_query:
                    logger.warning("spaCy fallback also resulted in an empty query. Using first 5 words of input as last resort.")
                    generated_query = ' '.join(text_input.split()[:5])

            except Exception as spacy_e:
                logger.error(f"Fallback spaCy keyword extraction also failed: {spacy_e}")
                generated_query = ' '.join(text_input.split()[:5]) # Ultimate fallback

        final_query = generated_query.strip()
        # Replace spaced hyphens (e.g., "t - shirt") with "t-shirt"
        final_query = re.sub(r'\s+-\s+', '-', final_query)

        if len(final_query.split()) > 15:
            final_query = ' '.join(final_query.split()[:15])
            logger.warning(f"Final query was still too long, truncated to: {final_query}")

        logger.info(f"Generated Google shopping query: '{final_query}'")
        return final_query

    def refine_results(self, query: str, products: List[Dict]) -> List[Dict]:
        """Main refinement pipeline: analyze query, filter, and rerank products."""
        if not query or not products:
            logger.warning("Refine_results called with empty query or products. Returning products as is.")
            return products

        try:
            criteria = self.analyze_query(query, products)
            filtered = self.apply_filters(products, criteria)
            final = self.rerank(filtered, criteria)
            logger.info(f"Final results: {len(final)} products")
            return final
        except Exception as e:
            logger.error(f"Error in refine_results: {str(e)}")
            return products
