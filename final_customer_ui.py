import os
import re
import json
import requests
import streamlit as st
import nest_asyncio
import asyncio
from dotenv import load_dotenv

from agents import (
    Agent,
    Runner,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    function_tool,
    ModelSettings,
)

# -----------------------------
# Fix for Streamlit + asyncio
# -----------------------------
nest_asyncio.apply()

# -----------------------------
# 1) Load env
# -----------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"  

# -----------------------------
# 2) API Setup
# -----------------------------
API_URL = "http://tfs.aesl.com.pk/web/fetch_customersx"

def fetch_customer_data(filters=None):
    """Fetch customer data from API with optional filters"""
    try:
        payload = {
            "jsonrpc": "2.0",
            "params": {
                "db_name": "04-Dec-2023"
            }
        }
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
            return data.get("result", [])
        else:
            st.error(f"API Error: {response.status_code}")
            return []
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return []

# -----------------------------
# 3) LLM Setup
# -----------------------------
external_client: AsyncOpenAI = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url=BASE_URL,
)

llm_model: OpenAIChatCompletionsModel = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=external_client
)

# -----------------------------
# 4) Tool: filter records (STANDALONE FUNCTION)
# -----------------------------
def filter_customers_manual(query: str, records: list, fields: list[str] = ("cus_id", "name", "region", "phone")):
    """Filter customer records based on query (cus_id, name, region)"""
    cus_id = None
    name = None
    region = None

    m_id = re.search(r"(?:\bcus[_\s-]?id\b|\bcustomer id\b|\bid\b)\s*[:\-]?\s*(\d+)", query, flags=re.I)
    if m_id:
        cus_id = m_id.group(1)

    m_name = re.search(r"(?:\bcustomer name\b|\bname\b)\s*[:\-]?\s*([A-Za-z0-9\-\(\)\s]+)", query, flags=re.I)
    if m_name:
        name = m_name.group(1).strip()

    m_region = re.search(r"\bregion\b\s*[:\-]?\s*([A-Za-z0-9\-\(\)\s]+)", query, flags=re.I)
    if m_region:
        region = m_region.group(1).strip()

    def matches(r):
        if cus_id and str(r.get("cus_id")) != str(cus_id):
            return False
        if name and name.lower() not in str(r.get("name","")).lower():
            return False
        if region and region.lower() not in str(r.get("region","")).lower():
            return False
        return True

    filtered = [r for r in (records or []) if matches(r)]

    items = []
    for r in filtered:
        items.append({f: r.get(f) for f in fields})

    return items

# -----------------------------
# 4b) Tool: filter records (FOR AGENT)
# -----------------------------
@function_tool
def filter_customer_records(
    query: str,
    records: list,
    fields: list[str] = ("cus_id","name","region","phone")
) -> list:
    """Filter customer records based on query (cus_id, name, region)"""
    # Use the manual function to avoid code duplication
    return filter_customers_manual(query, records, fields)

# -----------------------------
# 5) Customer Agent
# -----------------------------
customer_agent: Agent = Agent(
    name="Customer Agent",
    instructions=(
        "You are a customer-data specialist. The user will provide a query. "
        "You will receive `records` (list of customer dicts). "
        "Always call the tool `filter_customer_records` once, passing the query and records. "
        "Return the filtered results directly without any additional formatting."
    ),
    model=llm_model,
    tools=[filter_customer_records],
)

# -----------------------------
# 6) Streamlit UI
# -----------------------------
st.set_page_config(page_title="Customer Agent", layout="wide")

st.title("🔍 Customer Query Agent")

query = st.text_input("Enter your query:", "Show details of customer with cus_id 83907")

if st.button("Search"):
    with st.spinner("Fetching customer data..."):
        records = fetch_customer_data()

    if not records:
        st.warning("No customer data received from API.")
    else:
        st.write(f"Fetched {len(records)} records")
        
        # Option 1: Try with Agent first
        use_agent = st.checkbox("Use AI Agent (Experimental)", value=False)
        
        if use_agent:
            with st.spinner("Running Customer Agent..."):
                try:
                    out = asyncio.run(Runner.run(
                        customer_agent,
                        input={
                            "query": query,
                            "records": records
                        }
                    ))
                    
                    # Handle the output
                    if hasattr(out.final_output, '__iter__') and not isinstance(out.final_output, (str, dict)):
                        filtered_results = list(out.final_output)
                    else:
                        filtered_results = [out.final_output]
                        
                    st.subheader("AI Agent Results")
                    st.write(f"Found {len(filtered_results)} matching customers:")
                    
                except Exception as e:
                    st.error(f"Agent Error: {str(e)}")
                    st.info("Falling back to direct filtering...")
                    filtered_results = filter_customers_manual(query, records)
                    st.subheader("Direct Filtering Results")
                    st.write(f"Found {len(filtered_results)} matching customers:")
        else:
            # Option 2: Direct filtering (more reliable)
            filtered_results = filter_customers_manual(query, records)
            st.subheader("Filtering Results")
            st.write(f"Found {len(filtered_results)} matching customers:")
        
        # Display results
        if filtered_results:
            for i, customer in enumerate(filtered_results, 1):
                st.write(f"**Customer {i}:**")
                st.json(customer)
        else:
            st.info("No customers found matching your query.")
            
        # Show raw query for debugging
        with st.expander("Debug Info"):
            st.write("Query:", query)
            st.write("Total records:", len(records))
            st.write("Filtered count:", len(filtered_results))