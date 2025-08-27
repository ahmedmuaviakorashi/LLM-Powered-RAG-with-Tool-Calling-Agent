from langgraph.graph import StateGraph, END
from .base_agent import AgentState, LLMEnhancedReturnsAgent

class GraphBuilder:
    def __init__(self, agent: LLMEnhancedReturnsAgent):
        self.agent = agent
    
    def build_graph(self):
        builder = StateGraph(AgentState)
        
        # nodes
        builder.add_node("classify_intent", self.agent.classify_intent)
        builder.add_node("extract_parameters", self.agent.extract_parameters_llm)
        builder.add_node("perform_rag_search", self.agent.perform_rag_search)
        builder.add_node("compute_refund", self.agent.compute_refund)
        builder.add_node("generate_response", self.agent.generate_final_response)
        
        # edges
        builder.set_entry_point("classify_intent")
        
        # Route after intent classification
        def route_after_intent(state: AgentState):
            intent = state.get("intent", "")
            
            if intent == "rag_only":
                return "perform_rag_search"
            elif intent == "tool_only":
                return "extract_parameters"
            elif intent == "both":
                return "perform_rag_search"
            else:
                return "generate_response"
        
        builder.add_conditional_edges(
            "classify_intent",
            route_after_intent,
            {
                "perform_rag_search": "perform_rag_search",
                "extract_parameters": "extract_parameters",
                "generate_response": "generate_response"
            }
        )
        
        # Route after RAG search
        def route_after_rag(state: AgentState):
            intent = state.get("intent", "")
            
            if intent == "rag_only":
                return "generate_response"
            elif intent == "both":
                return "extract_parameters"
            else:
                return "generate_response"
        
        builder.add_conditional_edges(
            "perform_rag_search",
            route_after_rag,
            {
                "extract_parameters": "extract_parameters",
                "generate_response": "generate_response"
            }
        )
        
        # Route after parameter extraction
        def route_after_extraction(state: AgentState):
            missing_params = state.get("missing_params", [])
            
            if missing_params:
                return "generate_response"
            else:
                return "compute_refund"
        
        builder.add_conditional_edges(
            "extract_parameters",
            route_after_extraction,
            {
                "compute_refund": "compute_refund",
                "generate_response": "generate_response"
            }
        )
        
        # Route after computation
        builder.add_edge("compute_refund", "generate_response")
        
        # Final response
        builder.add_edge("generate_response", END)
        
        return builder.compile()