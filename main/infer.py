import pandas as pd
import pickle

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# =================================================
# IMPORTS
# =================================================
from src.config.settings import (
    FPGROWTH_RULE_INDEX_PATH
)



class ContextRecommender:
    def __init__(self, rules_path):
        """
        Initialize the recommender with the path to the pickle file containing context rules.
        """
        if not os.path.exists(rules_path):
            raise FileNotFoundError(f"Rules file not found at: {rules_path}")
            
        print(f"Loading rules from {rules_path}...")
        self.rules_df = pd.read_pickle(rules_path)
        
        # Pre-process for faster lookup
        # Convert lists to sets for fast subset operations
        self.rules_df['antecedent_set'] = self.rules_df['antecedent_products'].apply(lambda x: set(x) if isinstance(x, list) else set())
        self.rules_df['context_set'] = self.rules_df['context'].apply(lambda x: set(x) if isinstance(x, list) else set())
        
        print(f"Loaded {len(self.rules_df):,} rules.")
        
    def predict(self, cart_items, current_context, top_k=5):
        """
        Recommend products based on current cart and context.
        
        Args:
            cart_items (list): List of product names currently in the cart.
            current_context (list): List of context tags (e.g. ['Morning', 'Weekend']).
            top_k (int): Number of recommendations to return.
            
        Returns:
            list: List of dictionaries containing recommended products and metadata.
        """
        cart_set = set(cart_items)
        context_set = set(current_context)
        
        # 1. Context Filtering
        # Find rules where the rule's context requirement is met by the current context
        # rule.context <= current_context
        # Example: Rule needs [Weekend] -> Matches current [Weekend, Morning]
        context_mask = self.rules_df['context_set'].apply(lambda x: x.issubset(context_set))
        applicable_rules = self.rules_df[context_mask]
        
        if applicable_rules.empty:
            return []
            
        # 2. Cart Filtering
        # Find rules where the antecedent products are in the cart
        # rule.antecedent <= cart_items
        # Example: Rule needs [Banana] -> Matches cart [Banana, Apple]
        cart_mask = applicable_rules['antecedent_set'].apply(lambda x: x.issubset(cart_set))
        candidates = applicable_rules[cart_mask]
        
        if candidates.empty:
            return []
            
        # 3. Ranking
        # Sort by Lift Ratio (improvement over global) then Confidence
        # We might also want to prioritize longer antecedents (more specific matches)
        candidates = candidates.sort_values(by=['lift_ratio', 'conf'], ascending=False)
        
        # 4. Dedup and Format
        results = []
        seen = cart_set.copy() # Don't recommend what's already in the cart
        
        for _, row in candidates.iterrows():
            if len(results) >= top_k:
                break
            
            recs = row['recommended_products']
            # recs is a list, e.g. ['Apple']
            for item in recs:
                if item not in seen:
                    results.append({
                        'product': item,
                        'score': row['lift_ratio'],
                        'confidence': row['conf'],
                        'reason': f"Because you bought {row['antecedent_products']} in {row['context']}"
                    })
                    seen.add(item)
                    if len(results) >= top_k:
                        break
                        
        return results

if __name__ == "__main__":
    # Test block
    RULES_FILE = FPGROWTH_RULE_INDEX_PATH
    if os.path.exists(RULES_FILE):
        recommender = ContextRecommender(RULES_FILE)
        
        # Test Case 1
        cart = ['Organic Bananas']
        ctx = ['Morning', 'Weekend']
        print(f"\nTest 1: Cart={cart}, Context={ctx}")
        recs = recommender.predict(cart, ctx)
        for r in recs:
            print(f" - {r['product']} (Lift: {r['score']:.2f})")
            
        # Test Case 2 (Empty cart, pure context)
        cart = []
        ctx = ['Evening']
        print(f"\nTest 2: Cart={cart}, Context={ctx}")
        recs = recommender.predict(cart, ctx)
        for r in recs:
            print(f" - {r['product']} (Lift: {r['score']:.2f})")
