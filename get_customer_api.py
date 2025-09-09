import streamlit as st
import asyncio
import os
import requests  # Use requests instead of aiohttp
import json
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled, function_tool, ModelSettings, handoff, ItemHelpers
from tavily import AsyncTavilyClient
from openai.types.responses import ResponseTextDeltaEvent

# Load environment variables
load_dotenv(find_dotenv())

# Set page configuration
st.set_page_config(
    page_title="Allied Agent System",
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
if 'customer_data' not in st.session_state:
    st.session_state.customer_data = None

# API URL
API_URL = "http://tfs.aesl.com.pk/web/fetch_customersx"

# Function to fetch customer data from API (synchronous version)
# def fetch_customer_data(filters=None):
#     """Fetch customer data from API with optional filters"""
#     try:
#         response = requests.get(API_URL, params=filters, timeout=100)
#         if response.status_code == 200:
#             data = response.json()
#             return data
#         else:
#             st.error(f"API Error: {response.status_code}")
#             return None
#     except Exception as e:
#         st.error(f"Error fetching data: {e}")
#         return None

def fetch_customer_data(filters=None):
    """Fetch customer data from API with optional filters"""
    try:
        payload = {
            "jsonrpc": "2.0",
            "params": {
                "db_name": "04-Dec-2023"
            }
        }
        # agar filters diye hain (jaise cus_id, name) to params ke andar inject karo
        if filters:
            payload["params"].update(filters)

        response = requests.post(
            API_URL,
            auth=("devop@aesl.com.pk", "T3sT@696"),
            json=payload,
            timeout=60
        )
        if response.status_code == 200:
            data = response.json()
            return data.get("result")  # API ke format ke hisaab se "result" nikalna
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None


# # Function to extract customer info from prompt
# def extract_customer_info(prompt):
#     """Extract customer name or ID from prompt"""
#     prompt_lower = prompt.lower()
#     customer_info = {}
    
#     # Look for customer ID patterns
#     if any(word in prompt_lower for word in ["customer id", "cus_id", "id", "code","customer","name","region"]):
#         # Try to extract numeric ID
#         import re
#         numbers = re.findall(r'\d+', prompt)
#         if numbers:
#             customer_info['cus_id'] = numbers[0]
    
#     # Look for customer name patterns
#     if any(word in prompt_lower for word in ["customer name", "name", "customer"]):
#         # Try to extract name (this is a simple approach)
#         words = prompt.split()
#         for i, word in enumerate(words):
#             if word.lower() in ["customer", "name", "is"] and i + 1 < len(words):
#                 customer_info['name'] = words[i + 1]
#                 break
    
#     return customer_info

def extract_customer_info(prompt: str):
    """Extract customer filters (cus_id or name) from prompt"""
    prompt_lower = prompt.lower()
    customer_info = {}
    import re

    # Explicitly check for cus_id only when mentioned
    if "cus_id" in prompt_lower or "customer id" in prompt_lower or "id" in prompt_lower:
        numbers = re.findall(r'\d+', prompt)
        if numbers:
            customer_info['cus_id'] = numbers[0]

    # Extract customer name if explicitly mentioned
    if "name" in prompt_lower or "customer name" in prompt_lower:
        words = prompt.split()
        for i, word in enumerate(words):
            if word.lower() in ["customer", "name", "is"] and i + 1 < len(words):
                customer_info['name'] = words[i + 1]
                break

    return customer_info

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
st.title("Allied AI Agent System")
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
        placeholder="e.g., What are pros and cons of electric cars? or Fetch customer data for John",
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
            
            # Check if this is a customer data request
            if any(keyword in research_query.lower() for keyword in ["customer", "fetch customer", "user_id", "cus_id","region","name"]):
                st.session_state.customer_data = "pending"
        else:
            st.warning("Please enter a research question.")

with col2:
    st.header("Research Results")
    result_container = st.empty()
    result_container.write(st.session_state.research_result)
    
    # Display customer data if available
    # if st.session_state.customer_data and st.session_state.customer_data != "pending":
    #     st.subheader("Customer Data")
    #     if isinstance(st.session_state.customer_data, pd.DataFrame):
    #         st.dataframe(st.session_state.customer_data)
    #     elif isinstance(st.session_state.customer_data, list):
    #         st.dataframe(pd.DataFrame(st.session_state.customer_data))
    #     else:
    #         st.write(st.session_state.customer_data)

    # Display customer data (raw output)

    # customer_data = st.session_state.customer_data
    # if customer_data is not None:
    #     if isinstance(customer_data, str) and customer_data == "pending":
    #         pass
    #     else:
    #         st.subheader("Customer Data")
    #         st.json({"result": customer_data})   # wrap list into dict

customer_data = st.session_state.customer_data

if customer_data is not None:
    if isinstance(customer_data, str) and customer_data == "pending":
        pass
    else:
        st.subheader("Customer Data")

        # extract filters from user prompt
        filters = extract_customer_info(research_query)
        cus_id = filters.get("cus_id")
        st.write(cus_id)

        # actual records API response ke "result" key me hain
        records1 = customer_data
        st.write(records1)
        records = customer_data.get('result')
        st.write(records)
        if cus_id:
            # filter by cus_id
            filtered = [c for c in records1['cus_id'] if int(c) == int(cus_id)]
            st.write(filtered)
            if filtered:
                record = customer_data[filtered]
                st.subheader("RECORD Data")
                st.write(record)

                # sirf selected fields dikhana
                selected_fields = {
                    "cus_id": record.get('cus_id'),
                    "name": record.get('name'),
                    "region": record.get('region'),
                    "phone": record.get('phone'),
                }
                st.json(selected_fields)
            else:
                st.warning(f"No record found for cus_id {cus_id}")
        else:
            # agar user ne id nahi di to pura list show karo
            st.json(records)

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
            # Check if this is a customer data request
            if any(keyword in research_query.lower() for keyword in ["customer", "fetch customer", "user_id", "cus_id"]):
                st.session_state.research_result += "\n\n👥 Fetching customer data from API..."
                result_container.write(st.session_state.research_result)
                
                # Extract customer info from prompt
                customer_filters = extract_customer_info(research_query)
                
                # Fetch data from API (synchronous call)
                customer_data = fetch_customer_data(customer_filters)
                
                if customer_data:
                    # Convert to DataFrame for tabular display
                    if isinstance(customer_data, list):
                        df = pd.DataFrame(customer_data)
                        st.session_state.customer_data = df
                    else:
                        st.session_state.customer_data = customer_data
                    
                    st.session_state.research_result += f"\n\n✅ Retrieved {len(customer_data) if isinstance(customer_data, list) else 1} customer records"
                    result_container.write(st.session_state.research_result)
                else:
                    st.session_state.research_result += "\n\n❌ No customer data found"
                    result_container.write(st.session_state.research_result)
                
                st.session_state.research_in_progress = False
                st.rerun()
                return
            
            # Regular research process
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