"""
Embedding and Search Service for ShopSmarter
Handles image embedding generation and similarity search using CLIP and FAISS.
"""
import os
import time
import torch
import numpy as np
from PIL import Image
import open_clip
import faiss
from typing import List, Dict, Optional, Tuple
import requests
from io import BytesIO
import hashlib
import logging
import json
from open_clip import tokenize

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(
        self,
        model_name: str = None,
        pretrained: str = None,
        cache_dir: str = None,
        index_path: str = None
    ):
        """
        Initialize the embedding service with CLIP model and FAISS index.
        Args:
            model_name: CLIP model architecture (ENV: CLIP_MODEL or default "ViT-B-32")
            pretrained: Pretrained model weights (ENV: CLIP_PRETRAINED or default "laion2b_s34b_b79k")
            cache_dir: Directory to cache embeddings (ENV: EMBEDDING_CACHE_DIR or default "cache/embeddings")
            index_path: Base path for saving/loading index (ENV: FAISS_INDEX_PATH or default "cache/faiss_index")
        """
        # Configuration from environment
        self.model_name = model_name or os.getenv("CLIP_MODEL", "ViT-B-32")
        self.pretrained = pretrained or os.getenv("CLIP_PRETRAINED", "laion2b_s34b_b79k")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load CLIP model
        logger.info(
            f"Loading CLIP model {self.model_name}, pretrained={self.pretrained} on device={self.device}"
        )
        try:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                self.model_name,
                pretrained=self.pretrained,
                device=self.device
            )
            self.model.eval()  # Set to evaluation mode
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {str(e)}")
            raise

        # Determine embedding dimension dynamically
        try:
            test_dummy = torch.zeros(1, 3, 224, 224).to(self.device)
            with torch.no_grad():
                dim = self.model.encode_image(test_dummy).shape[-1]
            self.embedding_dim = int(dim)
        except Exception as e:
            logger.error(f"Failed to determine embedding dimension: {str(e)}")
            raise

        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.index_path = index_path or os.getenv("FAISS_INDEX_PATH", "cache/faiss_index")

        # Load candidate tag list
        cfg_path = os.getenv("TAGS_JSON")
        if not cfg_path:
            raise EnvironmentError("TAGS_JSON environment variable not set")
        with open(cfg_path, 'r') as f:
            self.candidates: List[str] = json.load(f)
        logger.info(f"Loaded {len(self.candidates)} candidate tags")

        # Cache for embeddings
        self.cache_dir = cache_dir or os.getenv("EMBEDDING_CACHE_DIR", "cache/embeddings")
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)

        # Product mapping: index position -> product dict
        self.product_map: Dict[int, Dict] = {}

    def load_image(self, image_path_or_url: str) -> Optional[Image.Image]:
        """
        Load an image from a local path or URL with retries and timeouts.
        """
        if not image_path_or_url:
            logger.error("Empty image path/URL provided")
            return None

        tries = 3
        for attempt in range(1, tries + 1):
            try:
                if image_path_or_url.startswith(('http://', 'https://')):
                    logger.debug(f"Fetching image URL (attempt {attempt}): {image_path_or_url}")
                    response = requests.get(image_path_or_url, timeout=10)
                    response.raise_for_status()
                    image = Image.open(BytesIO(response.content))
                else:
                    logger.debug(f"Loading local image: {image_path_or_url}")
                    image = Image.open(image_path_or_url)

                if image.mode != 'RGB':
                    image = image.convert('RGB')
                return image

            except requests.exceptions.RequestException as e:
                logger.warning(f"Network error loading image (attempt {attempt}): {str(e)}")
                if attempt < tries:
                    time.sleep(1 * attempt)
            except Exception as e:
                logger.error(f"Error loading image (attempt {attempt}): {str(e)}")
                if attempt < tries:
                    time.sleep(1 * attempt)

        logger.error(f"Failed to load image after {tries} attempts: {image_path_or_url}")
        return None

    def get_image_embedding(self, image: Image.Image) -> Optional[np.ndarray]:
        """
        Generate CLIP embedding for an image.
        """
        if image is None:
            return None

        try:
            tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            start = time.time()
            with torch.no_grad():
                features = self.model.encode_image(tensor)
            features /= features.norm(dim=-1, keepdim=True)
            embedding = features.cpu().numpy()
            logger.debug(f"Generated embedding in {time.time() - start:.3f}s")
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            return None
    
    def get_image_tags(self, image: Image.Image, top_k: int = 5) -> List[str]:
        """
        Zero-shot CLIP tag matching against self.candidates.
        """
        if not image:
            return []

        #Embed all candidate texts
        tokenized = tokenize(self.candidates).to(self.device)
        with torch.no_grad():
            text_feats = self.model.encode_text(tokenized)
            text_feats /= text_feats.norm(dim=-1, keepdim=True)

        #Embed the image
        img_emb = self.get_image_embedding(image)  # numpy array
        img_tensor = torch.from_numpy(img_emb).to(self.device)

        #Cosine-similarity
        sims = (img_tensor @ text_feats.T).cpu().numpy()[0]
        idxs = sims.argsort()[-top_k:][::-1]  # top_k indices
        tags = [self.candidates[i] for i in idxs]
        logger.info(f"Extracted image tags: {tags}")
        return tags

    def cache_embedding(self, product_id: str, embedding: np.ndarray):
        """Cache an embedding to disk."""
        if embedding is None:
            return

        try:
            path = os.path.join(self.cache_dir, f"{product_id}.npy")
            np.save(path, embedding)
            logger.debug(f"Cached embedding for {product_id} at {path}")
        except Exception as e:
            logger.error(f"Failed to cache embedding: {str(e)}")

    def load_cached_embedding(self, product_id: str) -> Optional[np.ndarray]:
        """Load a cached embedding from disk."""
        try:
            path = os.path.join(self.cache_dir, f"{product_id}.npy")
            if os.path.exists(path):
                logger.debug(f"Loading cached embedding for {product_id}")
                return np.load(path)
        except Exception as e:
            logger.error(f"Failed to load cached embedding: {str(e)}")
        return None

    def add_to_index(self, product: Dict):
        """
        Add a product's image embedding to the FAISS index, caching if needed.
        """
        if not product or not product.get("url") or not product.get("image_url"):
            logger.warning("Invalid product data, skipping")
            return

        try:
            product_id = hashlib.md5(product["url"].encode()).hexdigest()
            embedding = self.load_cached_embedding(product_id)
            
            if embedding is None:
                image = self.load_image(product["image_url"])
                if image is None:
                    return
                embedding = self.get_image_embedding(image)
                if embedding is None:
                    return
                self.cache_embedding(product_id, embedding)

            # Add embedding
            prev_count = self.index.ntotal
            self.index.add(embedding)
            logger.debug(f"Added embedding for {product_id}, index size {prev_count} -> {self.index.ntotal}")

            # Map index position to product
            self.product_map[self.index.ntotal - 1] = product
        except Exception as e:
            logger.error(f"Failed to add product to index: {str(e)}")

    def build_index(self, products: List[Dict]):
        """
        Build FAISS index from a list of products.
        """
        if not products:
            logger.warning("No products provided for indexing")
            return

        logger.info(f"Building index for {len(products)} products")
        try:
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.product_map.clear()
            for prod in products:
                self.add_to_index(prod)
            logger.info(f"Index built: total embeddings {self.index.ntotal}")
        except Exception as e:
            logger.error(f"Failed to build index: {str(e)}")
            raise

    def find_similar(
        self,
        query_image: Image.Image,
        k: int = 10,
        price_range: Optional[Tuple[float, float]] = None
    ) -> List[Dict]:
        """
        Find similar products to a query image.
        """
        if query_image is None:
            logger.error("No query image provided")
            return []

        if self.index.ntotal == 0:
            logger.warning("Index is empty")
            return []

        try:
            logger.info("Performing similarity search")
            query_emb = self.get_image_embedding(query_image)
            if query_emb is None:
                return []

            start = time.time()
            distances, indices = self.index.search(query_emb, min(k, self.index.ntotal))
            logger.debug(f"Search took {time.time() - start:.3f}s")

            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx not in self.product_map:
                    continue
                prod = dict(self.product_map[idx])
                prod["distance"] = float(dist)
                if price_range:
                    mn, mx = price_range
                    if not (mn <= prod.get("price", 0) <= mx):
                        continue
                results.append(prod)

            # Sort by ascending distance (most similar first)
            results = sorted(results, key=lambda x: x["distance"])
            logger.info(f"Returning {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Failed to find similar products: {str(e)}")
            return []

    def save_index(self):
        """Save FAISS index and product mapping to disk."""
        try:
            idx_path = f"{self.index_path}.index"
            json_path = f"{self.index_path}.json"
            faiss.write_index(self.index, idx_path)
            logger.info(f"FAISS index saved to {idx_path}")
            
            with open(json_path, 'w') as f:
                json.dump(self.product_map, f)
            logger.info(f"Product mapping saved to {json_path}")
        except Exception as e:
            logger.error(f"Failed to save index: {str(e)}")

    def load_index(self):
        """Load FAISS index and product mapping from disk."""
        try:
            idx_path = f"{self.index_path}.index"
            json_path = f"{self.index_path}.json"
            if os.path.exists(idx_path) and os.path.exists(json_path):
                self.index = faiss.read_index(idx_path)
                with open(json_path, 'r') as f:
                    self.product_map = json.load(f)
                logger.info(f"Loaded index and product map from disk: {len(self.product_map)} items")
            else:
                logger.warning(f"Index files not found at {self.index_path}, skipping load")
        except Exception as e:
            logger.error(f"Failed to load index: {str(e)}")
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.product_map.clear()
