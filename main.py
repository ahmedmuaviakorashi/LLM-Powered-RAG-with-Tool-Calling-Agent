import streamlit as st
import os
from dotenv import load_dotenv

from models.llm_providers import GroqProvider
from agents.base_agent import LLMEnhancedReturnsAgent
from agents.graph_builder import GraphBuilder
from langchain_core.messages import HumanMessage

load_dotenv()

# Streamlit Interface with Groq Integration
def main():
    st.set_page_config(
        page_title="Groq RAG Returns & Warranty Agent",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 LLM-Powered RAG & Tool Calling Agent")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Groq Configuration in sidebar
    with st.sidebar:
        st.header("🔧 Groq Configuration")
        
        try:
            import groq
            GROQ_AVAILABLE = True
        except ImportError:
            GROQ_AVAILABLE = False
            st.error("Groq package not installed. Run: pip install groq")
        
        if GROQ_AVAILABLE:
            # Get API key from environment or user input
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                api_key = st.text_input("Groq API Key:", type="password")
            
            model = st.selectbox(
                "Model:",
                ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768"],
                help="Select your preferred Groq model"
            )
            
            llm_provider = None
            
            if api_key:
                try:
                    llm_provider = GroqProvider(api_key, model)
                    st.success("✅ Groq configured")
                except Exception as e:
                    st.error(f"Error configuring Groq: {str(e)}")
            else:
                st.warning("Please enter your Groq API key")
        
        st.markdown("---")
        
        # Test cases
        st.header("🧪 Test Cases")
        test_cases = [
            "What's your return window for electronics?",
            "Do you charge a restocking fee for opened items?", 
            "I paid $300 for a sealed blender, delivered 10 days ago. How much refund?",
            "Headphones for $200, opened, delivered 12 days ago — refund?",
            "I bought a jacket last week for $120; how much can I get back?",
            "I'm past 35 days — can I still return?",
            "Return policy + estimate for a sealed phone $900, 14 days since delivery.",
            "I heard there's no restocking fee for electronics."
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            if st.button(f"Test {i}", key=f"test_{i}", help=test_case):
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": test_case})
                # Process the query
                process_query_with_ui(test_case, llm_provider)
        
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    # Initialize agent
    if llm_provider:
        if 'llm_agent' not in st.session_state or st.session_state.get('llm_provider') != model:
            agent = LLMEnhancedReturnsAgent(llm_provider)
            graph_builder = GraphBuilder(agent)
            st.session_state.llm_agent = agent
            st.session_state.agent_graph = graph_builder.build_graph()
            st.session_state.llm_provider = model
    else:
        st.warning("Configure Groq in the sidebar to use the agent")
        return
    
    # Chat input
    if prompt := st.chat_input("Ask about returns or get refund estimates:"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Process the query
        process_query_with_ui(prompt, llm_provider)
        
# Process a query and update the UI with the response
def process_query_with_ui(query: str, llm_provider):
    if 'llm_agent' not in st.session_state:
        agent = LLMEnhancedReturnsAgent(llm_provider)
        graph_builder = GraphBuilder(agent)
        st.session_state.llm_agent = agent
        st.session_state.agent_graph = graph_builder.build_graph()
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(query)
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        
        # Process the query
        with st.spinner("AI is processing your request..."):
            try:
                # Initialize state
                initial_state = {
                    "messages": [HumanMessage(content=query)],
                    "user_query": query,
                    "intent": "",
                    "extracted_params": {},
                    "rag_results": [],
                    "tool_result": {},
                    "missing_params": [],
                    "final_answer": "",
                    "citations": []
                }
                
                # Execute the graph
                final_state = st.session_state.agent_graph.invoke(initial_state)
                
                # Format the result
                result = {
                    'user_query': query,
                    'intent': final_state.get('intent', ''),
                    'extracted_params': final_state.get('extracted_params', {}),
                    'rag_results': final_state.get('rag_results', []),
                    'tool_result': final_state.get('tool_result', {}),
                    'missing_params': final_state.get('missing_params', []),
                    'final_answer': final_state.get('final_answer', ''),
                    'citations': final_state.get('citations', [])
                }
                
                # Display the final answer
                message_placeholder.markdown(result['final_answer'])
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": result['final_answer']})
                
                # Debug information
                with st.expander("🔍 AI Processing Details"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Intent Classification:**", result.get('intent', 'N/A'))
                        st.write("**Extracted Parameters:**")
                        st.json(result.get('extracted_params', {}))
                        
                        if result.get('citations'):
                            st.write("**Citations:**")
                            for citation in result['citations']:
                                st.write(f"- {citation}")
                    
                    with col2:
                        if result.get('rag_results'):
                            st.write("**RAG Results:**")
                            for i, rag_result in enumerate(result['rag_results'][:2]):
                                policy = rag_result['policy']
                                st.write(f"**{policy['title']}** (Score: {rag_result['score']:.3f})")
                                st.write(f"_{policy['content']}_")
                        
                        if result.get('tool_result'):
                            st.write("**Refund Calculation:**")
                            st.json(result['tool_result'])
                
            except Exception as e:
                error_msg = f"I apologize, but I encountered an error processing your request: {str(e)}"
                message_placeholder.markdown(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()