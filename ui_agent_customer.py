import os
import asyncio
import re
import requests
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Streamlit
import matplotlib.pyplot as plt
import io
import base64
import streamlit as st
from dotenv import load_dotenv
from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, set_tracing_disabled, function_tool

# 🌿 Load environment variables
load_dotenv()
set_tracing_disabled(disabled=True)

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

# 🔐 Setup Gemini client
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

external_client = AsyncOpenAI(api_key=GEMINI_API_KEY, base_url=BASE_URL)
llm_model = OpenAIChatCompletionsModel(model="gemini-2.5-flash", openai_client=external_client)

API_URL = "http://tfs.aesl.com.pk/web/fetch_customersx"

def fetch_customer_data(filters=None):
    """Dummy static data (replace with fetch_customer_data1 for live API)"""
    DATA = [
        {"user_id": 1001, "name": "ZUBAIR HASSAN", "cus_id": "83907", "email": "ali.ahmed@email.com", "phone": "0300-1234567", "city": "Lahore"},
        {"user_id": 1002, "name": "(FWO) PD HQ NBBIAP", "cus_id": "88594", "email": "fatima.khan@email.com", "phone": "0312-9876543", "city": "Karachi"},
        {"user_id": 1003, "name": "156 INDEPENDENT INFANTRY", "cus_id": "81155", "email": "usman.malik@email.com", "phone": "0333-4567890", "city": "Islamabad"},
        {"user_id": 1009, "name": "Hina Sheikh", "cus_id": "82620", "email": "hina.sheikh@email.com", "phone": "0322-6543210", "city": "Sialkot"},
        {"user_id": 1010, "name": "A A BROTHER", "cus_id": "88737", "email": "omar.farooq@email.com", "phone": "0305-1357924", "city": "Sialkot"}
    ]
    return DATA

def filter_customers_manual(query: str, records: list, fields: list[str] = ("cus_id", "name", "city", "phone")):
    """Filter customer records manually"""
    region = None
    m_region = re.search(r"\b([A-Za-z]+)\s+customers?\b", query, flags=re.I)
    if m_region:
        region = m_region.group(1).strip()

    def matches(r):
        if region and region.lower() not in str(r.get("city","")).lower():
            return False
        return True

    filtered = [r for r in (records or []) if matches(r)]
    return [{f: r.get(f) for f in fields} for r in filtered]

@function_tool
def fetch_customer_data_tool() -> list:
    return fetch_customer_data()

# @function_tool
# def plot_customers(query: str, records: list) -> str:
#     """
#     Generate a bar chart of customers count per region or other criteria.
#     Returns a base64 image string so it can be displayed.
#     """

#     # Region-wise count
#     region_counts = {}
#     for r in records:
#         region = r.get("region", "Unknown")
#         region_counts[region] = region_counts.get(region, 0) + 1

#     # Plot
#     fig, ax = plt.subplots(figsize=(6, 4))
#     ax.bar(region_counts.keys(), region_counts.values(), color="skyblue")
#     ax.set_title("Customer Count per Region")
#     ax.set_xlabel("Region")
#     ax.set_ylabel("Number of Customers")
#     plt.xticks(rotation=45)

#     # Convert to base64 for return
#     # buf = io.BytesIO()
#     plt.tight_layout()
#     # plt.savefig(buf, format="png")
#     # buf.seek(0)
#     # img_base64 = base64.b64encode(buf.read()).decode("utf-8")
#     # plt.close(fig)
#     return fig

@function_tool
def plot_customers(query: str, records: list):
    """
    Generate a bar chart of customers count per region.
    Returns a Plotly figure object.
    """
    import plotly.express as px
    import pandas as pd
    
    region_counts = {}
    for r in records:
        region = r.get("region", "Unknown")
        region_counts[region] = region_counts.get(region, 0) + 1

    df = pd.DataFrame({
        'Region': list(region_counts.keys()),
        'Count': list(region_counts.values())
    })
    
    fig = px.bar(df, x='Region', y='Count', 
                 title='Customer Count per Region',
                 color='Count', color_continuous_scale='blues')
    fig.update_layout(xaxis_tickangle=-45)
    
    return fig

@function_tool
def count_customers(query: str, records: list) -> dict:
    region = None
    m_region = re.search(r"\b([A-Za-z]+)\s+customers?\b", query, flags=re.I)
    if m_region:
        region = m_region.group(1).strip()
    filtered = records
    if region:
        filtered = [r for r in records if region.lower() in str(r.get("city","")).lower()]
    return {"total": len(filtered), "region": region if region else "all"}

@function_tool
def filter_customer_records(query: str, records: list, fields: list[str] = ("cus_id","name","city","phone")) -> list:
    return filter_customers_manual(query, records, fields)

# Agent setup
customer_agent: Agent = Agent(
    name="Customer Agent",
    instructions=(
        "You are a customer-data specialist. "
        "If the query asks about visualization, charts, or graphs, call `plot_customers`. "
        "If it's about counting, call `count_customers`. "
        "If it's about filtering, call `filter_customer_records`. "
        "Always use `fetch_customer_data_tool` first to get records."
    ),
    model=llm_model,
    tools=[fetch_customer_data_tool, filter_customer_records, count_customers,plot_customers],
)
async def run_agentic_query():
    result = await Runner.run(customer_agent, query)
    return result.final_output

# -------- Streamlit UI --------
st.title("📊 Customer Data Query (Agentic AI)")
st.write("Ask any query related to customers.")

query = st.text_input("Enter your query:", "Please display Sialkot customers in tabular format.")

if st.button("Run Query"):
    # async def run_agentic_query():
    #     result = await Runner.run(customer_agent, query)
    #     return result.final_output

    # result = asyncio.run(run_agentic_query())
    with st.spinner("⏳ Agent is running... Please wait."):
    # asyncio wrapper
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(run_agentic_query())
        finally:
            loop.close()

    st.subheader("🔍 Agent Output")
    if isinstance(result, plt.Figure):
        st.pyplot(result, use_container_width=True)  # Added use_container_width
    elif hasattr(result, '__class__') and 'plotly' in str(type(result)):
        st.plotly_chart(result, use_container_width=True)
    elif isinstance(result, list):
        st.dataframe(result, use_container_width=True)
    else:
        st.write(result)