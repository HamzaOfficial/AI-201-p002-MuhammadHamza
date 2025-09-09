import streamlit as st
import asyncio
import os
from dotenv import load_dotenv, find_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled, function_tool, ModelSettings, handoff, ItemHelpers
from tavily import AsyncTavilyClient
from openai.types.responses import ResponseTextDeltaEvent

# Load environment variables
load_dotenv(find_dotenv())

# Set page configuration
st.set_page_config(
    page_title="Research Agent System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if 'research_result' not in st.session_state:
    st.session_state.research_result = ""
if 'research_in_progress' not in st.session_state:
    st.session_state.research_in_progress = False
if 'query_history' not in st.session_state:
    st.session_state.query_history = []

# Initialize agents (will be done once)
@st.cache_resource
def initialize_agents():
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    tavily_api_key = os.environ.get("TAVILY_API_KEY")
    tavily_client = AsyncTavilyClient(api_key=tavily_api_key)
    
    # Tracing disabled
    set_tracing_disabled(disabled=True)

    # Set up LLM service
    external_client = AsyncOpenAI(
        api_key=gemini_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    # Set up LLM model
    llm_model = OpenAIChatCompletionsModel(
        model="gemini-2.5-flash",
        openai_client=external_client
    )

    # Define tools
    @function_tool
    async def search(query: str) -> str:
        """Smart and efficient searching function"""
        st.session_state.research_result += f"\n\n🔍 Searching for: {query}"
        result_container.write(st.session_state.research_result)
        response = await tavily_client.search(query)
        return response

    @function_tool
    async def extract_content(urls: list) -> dict:
        """An extracting urls function"""
        st.session_state.research_result += f"\n\n📥 Extracting content from: {', '.join(urls)}"
        result_container.write(st.session_state.research_result)
        response = tavily_client.extract(urls)
        return response

    # Define agents
    fact_finder = Agent(
        name="Fact Finder",
        model=llm_model,
        tools=[search, extract_content],
        instructions="You are a fact-finding researcher. Gather accurate data and provide references.",
        model_settings=ModelSettings(temperature=0.7),
    )

    source_checker = Agent(
        name="Source Checker",
        model=llm_model,
        instructions="You are responsible for validating sources. Check reliability, trustworthiness, and relevance.",
        model_settings=ModelSettings(temperature=0.5),
    )

    summary_writer = Agent(
        name="Summary Writer",
        model=llm_model,
        instructions="You are a research assistant. Write concise, clear, and well-structured summaries from the provided information.",
        model_settings=ModelSettings(temperature=0.9),
    )

    research_team = Agent(
        name="Research Team",
        model=llm_model,
        instructions=(
            "Coordinate the research workflow:\n"
            "1. Handoff to the Fact Finder to gather info.\n"
            "2. Handoff to the Source Checker to validate credibility.\n"
            "3. Handoff to the Summary Writer to produce the final report."
        ),
        handoffs=[
            handoff(fact_finder),
            handoff(source_checker),
            handoff(summary_writer),
        ]
    )
    
    return research_team, tavily_client

# UI Layout
st.title("🔍 Research Agent System")
st.markdown("Use AI agents to research topics and generate well-sourced summaries.")

# Sidebar
with st.sidebar:
    st.header("Settings")
    st.info("This system uses multiple AI agents to research topics, verify sources, and generate summaries.")
    
    st.divider()
    st.header("Query History")
    for i, query in enumerate(st.session_state.query_history):
        if st.button(f"{query}", key=f"history_{i}", use_container_width=True):
            st.session_state.research_query = query

# Main content area
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Research Query")
    research_query = st.text_input(
        "Enter your research question:",
        placeholder="e.g., What are pros and cons of electric cars?",
        key="research_query_input"
    )
    
    if st.button("Start Research", type="primary", disabled=st.session_state.research_in_progress):
        if research_query:
            st.session_state.research_in_progress = True
            st.session_state.research_result = "🔄 Starting research process...\n"
            st.session_state.query_history.append(research_query)
            # Limit history to 10 items
            if len(st.session_state.query_history) > 10:
                st.session_state.query_history = st.session_state.query_history[-10:]
        else:
            st.warning("Please enter a research question.")

with col2:
    st.header("Research Results")
    result_container = st.empty()
    result_container.write(st.session_state.research_result)

# Initialize agents
try:
    research_team, tavily_client = initialize_agents()
except Exception as e:
    st.error(f"Error initializing agents: {e}")
    st.stop()

# Run research if requested
if st.session_state.research_in_progress:
    async def run_research():
        try:
            st.session_state.research_result += "\n\n🧠 Research team is working on your query..."
            result_container.write(st.session_state.research_result)
            
            result = Runner.run_streamed(
                research_team,
                research_query
            )
            
            full_response = ""
            async for event in result.stream_events():
                if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                    delta = event.data.delta
                    full_response += delta
                    st.session_state.research_result = f"# Research Results\n\n**Query:** {research_query}\n\n**Response:**\n\n{full_response}"
                    result_container.write(st.session_state.research_result)
            
            st.session_state.research_result += "\n\n✅ Research completed!"
            result_container.write(st.session_state.research_result)
            
        except Exception as e:
            st.session_state.research_result += f"\n\n❌ Error during research: {e}"
            result_container.write(st.session_state.research_result)
        finally:
            st.session_state.research_in_progress = False
            st.rerun()
    
    # Run the async function
    asyncio.run(run_research())

# Footer
st.divider()
st.caption("Powered by Gemini AI and Tavily Search API")