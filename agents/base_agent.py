import json
import re
from typing import Dict, List, Any, TypedDict, Annotated
from langgraph.graph.message import add_messages
from models.llm_providers import LLMProvider
from models.vector_rag import VectorRAG
from tools.refund_calculator import RefundCalculator
from utils.helpers import load_json_file, extract_parameters_regex

# state for our agent
class AgentState(TypedDict):
    messages: Annotated[List[Any], add_messages]
    user_query: str
    intent: str
    extracted_params: Dict[str, Any]
    rag_results: List[Dict]
    tool_result: Dict[str, Any]
    missing_params: List[str]
    final_answer: str
    citations: List[str]

# LangGraph integration
class LLMEnhancedReturnsAgent:
    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider
        self.policies = load_json_file("data/policies.json")
        self.vector_rag = VectorRAG(llm_provider, self.policies)
        self.refund_calculator = RefundCalculator()
    
    # LLM prompt to classify user intent
    def classify_intent(self, state: AgentState) -> AgentState:
        query = state["user_query"]
        
        prompt = f"""
        Classify this customer query into exactly one category:

        1. "rag_only" - Query only asks about general policies without specific item details
           Examples: "What's your return policy?", "Do you charge restocking fees?"
        
        2. "tool_only" - Query has ALL required info for refund calculation (price, timeframe, condition, category)
           Examples: "$300 sealed blender, 10 days ago", "headphones $200 opened 12 days"
        
        3. "both" - Query mentions specific timeframes/conditions but needs both policy info AND may need follow-up questions
           Examples: "jacket $120 last week", "I'm past 35 days", "return policy + estimate for phone $900"

        Query: "{query}"

        Respond with only one word: rag_only, tool_only, or both
        """
        
        response = self.llm_provider.generate_response(prompt, max_tokens=10)
        intent = response.strip().lower().replace("_", "_")
        
        if intent not in ["rag_only", "tool_only", "both"]:
            intent = "both"
            
        state["intent"] = intent
        return state
    
    # extracting parameters
    def extract_parameters_llm(self, state: AgentState) -> AgentState:
        query = state["user_query"]
        
        prompt = f"""
        Extract information from this customer query. Return "unknown" if not provided:
        
        Query: "{query}"
        
        Extract these exact fields:
        1. purchase_price: Extract number only (no $ sign). Examples: 300, 120.50
        2. days_since_delivery: Convert to days. "yesterday"=1, "last week"=7, "12 days ago"=12
        3. opened: "opened" if item was opened/used, "sealed" if new/unopened, "unknown" if unclear
        4. category: "electronics", "apparel", "books", "home", or "unknown"
        
        Format as JSON only:
        {{"purchase_price": "unknown", "days_since_delivery": "unknown", "opened": "unknown", "category": "unknown"}}
        """
        
        try:
            response = self.llm_provider.generate_response(prompt, max_tokens=100)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                extracted = json.loads(json_match.group())
            else:
                extracted = json.loads(response)
            
            params = {}
            if extracted.get("purchase_price") != "unknown":
                try:
                    params["purchase_price"] = float(extracted["purchase_price"])
                except:
                    pass
            if extracted.get("days_since_delivery") != "unknown":
                try:
                    params["days_since_delivery"] = int(extracted["days_since_delivery"])
                except:
                    pass
            if extracted.get("opened") != "unknown":
                params["opened"] = extracted["opened"].lower() == "opened"
            if extracted.get("category") != "unknown":
                params["category"] = extracted["category"]
            
            state["extracted_params"] = params
            
            required_params = ['purchase_price', 'days_since_delivery', 'opened', 'category']
            missing = [p for p in required_params if p not in state["extracted_params"]]
            state["missing_params"] = missing
            
        except Exception as e:
            state = self.extract_parameters_regex(state)
        return state

    # regex-based parameter extraction
    def extract_parameters_regex(self, state: AgentState) -> AgentState:
        query = state["user_query"]
        params = extract_parameters_regex(query)
        
        state["extracted_params"] = params
        
        required_params = ['purchase_price', 'days_since_delivery', 'opened', 'category']
        missing = [p for p in required_params if p not in state["extracted_params"]]
        state["missing_params"] = missing
        
        return state
    
    # refund based on policy and parameters
    def compute_refund(self, state: AgentState) -> AgentState:
        params = state["extracted_params"]
        result = self.refund_calculator.compute_refund(params)
        state["tool_result"] = result
        return state
    # Perform RAG search and update state
    def perform_rag_search(self, state: AgentState) -> AgentState:
        query = state["user_query"]
        state["rag_results"] = self.vector_rag.semantic_search(query)
        return state
    
    # final response following the required format
    def generate_final_response(self, state: AgentState) -> AgentState:
        query = state["user_query"]
        intent = state.get("intent", "")
        rag_results = state.get("rag_results", [])
        tool_result = state.get("tool_result", {})
        missing_params = state.get("missing_params", [])
        
        if missing_params and intent in ["tool_only", "both"]:
            missing_param = missing_params[0]
            if missing_param == "opened":
                response = "Was the item opened or is it still sealed?"
            elif missing_param == "category":
                response = "What type of item is this? (electronics, apparel, books, or home goods)"
            elif missing_param == "purchase_price":
                response = "What was the original purchase price?"
            elif missing_param == "days_since_delivery":
                response = "How many days ago was it delivered?"
            else:
                response = f"I need to know: {missing_param.replace('_', ' ')}"
            
            state["final_answer"] = response
            return state
        
        response_parts = []
        used_components = []
        citations = []
        
        if rag_results and intent in ["rag_only", "both"]:
            top_policy = rag_results[0]['policy']
            citations.append(f"{top_policy['title']} (ID: {top_policy['id']})")
            used_components.append(f"policy '{top_policy['title']}'")
            
            if intent == "rag_only":
                response_parts.append(top_policy['content'])
            elif intent == "both":
                response_parts.append(f"According to our policy: {top_policy['content']}")
        
        if tool_result and tool_result.get('refund_amount') is not None:
            used_components.append("refund calculator")
            
            refund_amount = tool_result['refund_amount']
            applied_rules = tool_result.get('applied_rules', [])
            
            if refund_amount > 0:
                response_parts.append(f"Your refund would be ${refund_amount:.2f}")
                if applied_rules:
                    response_parts.append(f"Applied rules: {', '.join(applied_rules)}")
            else:
                response_parts.append("Unfortunately, no refund is available due to being past the return window.")
        
        query_lower = query.lower()
        has_timeframe = any(phrase in query_lower for phrase in ['days', 'week', 'month', 'past', 'ago', 'since'])
        has_item_mention = any(phrase in query_lower for phrase in ['phone', 'laptop', 'headphone', 'jacket', 'shirt', 'blender', 'book'])
        
        if intent == "both" and has_timeframe and not has_item_mention and not tool_result:
            if response_parts:
                response_parts.append("What type of item are you looking to return?")
            else:
                response_parts.append("What type of item are you looking to return?")
        
        if response_parts:
            response = ". ".join(response_parts) + "."
        else:
            response = "I can help you with returns and refund calculations. What would you like to know?"
        
        if used_components:
            response += f"\n\nWhat I used: {' + '.join(used_components)}"
        state["citations"] = citations
        state["final_answer"] = response
        return state