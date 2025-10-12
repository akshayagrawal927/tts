from .config import TOP_N_MATCHES_INT,FUZZY_SEARCH_THRESHOLD_FLOAT
from typing import List, Tuple, Literal
from thefuzz import fuzz
import logging

logger = logging.getLogger(__name__)

class FuzzySearch:
    """A system for performing fuzzy matching on product and customer names."""
    def __init__(self, products: List[str], customers: List[str], threshold: float = FUZZY_SEARCH_THRESHOLD_FLOAT):
        self.products = products
        self.customers = customers
        self.threshold = threshold
        logger.info(f"FuzzySearch initialized with {len(products)} products and {len(customers)} customers.")

    def find_best_matches(self, query: str, entity_type: Literal["product", "customer"]) -> List[Tuple[str, float]]:
        """Find the best matching items for a query using a more advanced ratio."""
        items = self.products if entity_type == "product" else self.customers
        query_lower = query.lower()
        matches = []

        for item in items:
            # Use token_set_ratio, which is excellent for matching strings with different word counts
            # e.g., "coke sleek can" vs "coke 330ml 1x24 np sleek can"
            # It finds the common tokens and compares them.
            similarity_score = fuzz.token_set_ratio(query_lower, item.lower())

            # Convert to a 0-1 float scale
            similarity_float = similarity_score / 100.0

            if similarity_float >= self.threshold:
                matches.append((item, similarity_float))

        # Sort by similarity score (descending)
        matches.sort(key=lambda x: x[1], reverse=True)

        # Return top N matches for better suggestions
        return matches[:TOP_N_MATCHES_INT]

    # def _similarity_ratio(self, s1: str, s2: str) -> float:
    #     """Calculate similarity ratio with improved exact matching."""
    #     s1, s2 = s1.lower().strip(), s2.lower().strip()
        
    #     # Perfect exact match
    #     if s1 == s2:
    #         return 1.0
        
    #     # Check for exact substring matches (for longer strings)
    #     if len(s1) > 4 and len(s2) > 4:
    #         if s1 in s2:
    #             return 0.98  # Very high score for exact substring
    #         if s2 in s1:
    #             return 0.98
        
    #     # Standard similarity using SequenceMatcher
    #     base_similarity = SequenceMatcher(None, s1, s2).ratio()
        
    #     # Word-level matching for multi-word entities
    #     words1 = s1.split()
    #     words2 = s2.split()
        
    #     if len(words1) > 1 or len(words2) > 1:
    #         # Count matching words
    #         common_words = 0
    #         total_words = max(len(words1), len(words2))
            
    #         for w1 in words1:
    #             for w2 in words2:
    #                 if w1 == w2:  # Exact word match
    #                     common_words += 1
    #                     break
    #                 elif len(w1) > 3 and len(w2) > 3 and (w1 in w2 or w2 in w1):
    #                     common_words += 0.8
    #                     break
    #                 elif SequenceMatcher(None, w1, w2).ratio() > 0.85:
    #                     common_words += 0.6
    #                     break
            
    #         word_similarity = common_words / total_words
    #         return max(base_similarity, word_similarity)
        
    #     return base_similarity

    # def find_best_matches(self, query: str, entity_type: Literal["product", "customer"]) -> List[Tuple[str, float]]:
    #     """Find the best matching items for a query."""
    #     items = self.products if entity_type == "product" else self.customers
    #     matches = []

    #     # Use a lower threshold for initial matching
    #     threshold = 0.2  

    #     for item in items:
    #         similarity = self._similarity_ratio(query, item)
    #         if similarity >= threshold:
    #             matches.append((item, similarity))
                
    #     # Sort by similarity score (descending)
    #     matches.sort(key=lambda x: x[1], reverse=True)
        
    #     # Return top 5 matches for better suggestions
    #     return matches[:5]
