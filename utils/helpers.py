import json
import re
from typing import Dict, Any


# Load JSON data from a file
def load_json_file(file_path: str) -> Any:
    with open(file_path, 'r') as file:
        return json.load(file)
# Fallback regex-based parameter extraction
def extract_parameters_regex(query: str) -> Dict[str, Any]:
    params = {}
    # Extract price
    price_match = re.search(r'\$?(\d+(?:\.\d{2})?)', query)
    if price_match:
        params['purchase_price'] = float(price_match.group(1))
    
    # Extract days since delivery
    if 'yesterday' in query.lower():
        params['days_since_delivery'] = 1
    elif re.search(r'(\d+)\s*days?\s*ago', query):
        days = int(re.search(r'(\d+)\s*days?\s*ago', query).group(1))
        params['days_since_delivery'] = days
    elif re.search(r'last\s*week', query.lower()):
        params['days_since_delivery'] = 7
    elif re.search(r'(\d+)\s*days?\s*(since|from)', query):
        days = int(re.search(r'(\d+)\s*days?\s*(since|from)', query).group(1))
        params['days_since_delivery'] = days
    
    # Extract condition
    if re.search(r'opened?|used', query.lower()):
        params['opened'] = True
    elif re.search(r'sealed|new|unopened', query.lower()):
        params['opened'] = False
    
    # Extract category with better matching
    categories = {
        'electronics': ['phone', 'laptop', 'headphone', 'headphones', 'tablet', 'computer', 'electronics', 'electronic'],
        'apparel': ['shirt', 'jacket', 'dress', 'shoes', 'clothes', 'apparel', 'clothing'],
        'books': ['book', 'dvd', 'cd', 'media'],
        'home': ['blender', 'kitchen', 'appliance', 'furniture', 'home']
    }
    
    for category, keywords in categories.items():
        if any(keyword in query.lower() for keyword in keywords):
            params['category'] = category
            break
    return params