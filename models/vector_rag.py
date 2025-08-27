import re
from typing import Dict, List
from .llm_providers import LLMProvider

class VectorRAG:
    def __init__(self, llm_provider: LLMProvider, policies: List[Dict]):
        self.llm_provider = llm_provider
        self.policies = policies
        self.build_keyword_index()

    # Build keyword index for search
    def build_keyword_index(self):
        self.keyword_index = {}
        for policy in self.policies:
            text = f"{policy['title']} {policy['content']}".lower()
            words = re.findall(r'\b\w+\b', text)
            for word in words:
                if word not in self.keyword_index:
                    self.keyword_index[word] = []
                if policy['id'] not in [p['id'] for p in self.keyword_index[word]]:
                    self.keyword_index[word].append(policy)
    # semantic search with better category detection
    def semantic_search(self, query: str, top_k: int = 3) -> List[Dict]:
        results = self.keyword_search(query, top_k)
        
        if not results or results[0]['score'] < 3:
            try:
                prompt = f"""
                Analyze this customer query and extract the key terms for policy search:
                Query: "{query}"
                
                Focus on:
                1. Item category (electronics, apparel, books, home)
                2. Policy type (return window, restocking fee, warranty)
                3. Key conditions (opened, sealed, damaged)
                
                Respond with only the most relevant keywords separated by commas:
                """
                
                response = self.llm_provider.generate_response(prompt, max_tokens=50)
                keywords = [kw.strip().lower() for kw in response.split(",")]
                enhanced_query = " ".join(keywords)
                enhanced_results = self.keyword_search(enhanced_query, top_k)
                
                if enhanced_results and enhanced_results[0]['score'] > (results[0]['score'] if results else 0):
                    results = enhanced_results  
            except:
                pass
        return results
    # Enhanced keyword search with category-aware scoring
    def keyword_search(self, query: str, top_k: int = 3) -> List[Dict]:
        results = []
        query_lower = query.lower()
        query_words = re.findall(r'\b\w+\b', query_lower)
        
        detected_category = None
        category_keywords = {
            'electronics': ['phone', 'laptop', 'headphone', 'headphones', 'tablet', 'computer', 'electronics', 'electronic'],
            'apparel': ['shirt', 'jacket', 'dress', 'shoes', 'clothes', 'apparel', 'clothing'],
            'books': ['book', 'dvd', 'cd', 'media'],
            'home': ['blender', 'kitchen', 'appliance', 'furniture', 'home']
        }
        
        for category, keywords in category_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_category = category
                break
        
        for policy in self.policies:
            content_lower = policy['content'].lower()
            title_lower = policy['title'].lower()
            policy_id = policy['id'].lower()
            
            score = 0
            
            for word in query_words:
                if word in content_lower:
                    score += 2
                if word in title_lower:
                    score += 3
            
            if detected_category:
                if detected_category == 'electronics':
                    if 'electronics' in policy_id or 'electronics' in title_lower:
                        score += 10
                    elif 'general' in policy_id and score > 0:
                        score += 1
                elif detected_category == 'apparel':
                    if 'apparel' in policy_id or 'apparel' in title_lower:
                        score += 10
                    elif 'general' in policy_id and score > 0:
                        score += 1
                elif detected_category == 'books':
                    if 'books' in title_lower or 'media' in title_lower:
                        score += 10
                    elif 'restocking' in policy_id and 'books' in content_lower:
                        score += 8
                elif detected_category == 'home':
                    if 'general' in policy_id and score > 0:
                        score += 5
            
            if any(word in query_lower for word in ['restocking', 'fee', 'charge', 'opened', 'sealed']):
                if 'restocking' in policy_id:
                    score += 8
            
            if any(word in query_lower for word in ['window', 'return', 'days']):
                if 'return' in policy_id and detected_category:
                    if detected_category in policy_id or detected_category in title_lower:
                        score += 8
            
            if score > 0:
                results.append({
                    'policy': policy,
                    'score': score
                }) 
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]