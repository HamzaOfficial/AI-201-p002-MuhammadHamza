import os
import asyncio
from dotenv import load_dotenv
import requests
import re
from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, set_tracing_disabled, ModelSettings, function_tool

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

def fetch_customer_data1(filters=None):
    """Fetch customer data from API with optional filters"""
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
        # print("Data Found --> ",data.get("result", []))
        return data.get("result", [])
    else:
        return []

def fetch_customer_data(filters=None):
    DATA = [
    {"user_id": 1001, "name": "ZUBAIR HASSAN (B12,COLONY3 CHAKLALA GARRISON RAWALPINDI)", "cus_id": "83907", "email": "ali.ahmed@email.com", "phone": "0300-1234567", "city": "Lahore"},
    {"user_id": 1002, "name": "(FWO) PD HQ NBBIAP", "cus_id": "88594", "email": "fatima.khan@email.com", "phone": "0312-9876543", "city": "Karachi"},
    {"user_id": 1003, "name": "156 INDEPENDENT INFANTRY WORKSHOP COMPANY", "cus_id": "81155", "email": "usman.malik@email.com", "phone": "0333-4567890", "city": "Islamabad"},
    {"user_id": 1004, "name": "1st for Connect Pvt Ltd", "cus_id": "83025", "email": "ayesha.raza@email.com", "phone": "0345-1122334", "city": "Rawalpindi"},
    {"user_id": 1005, "name": "3G TECHNOLOGIES (GURANTEE) LIMITED", "cus_id": "82989", "email": "bilal.hassan@email.com", "phone": "0321-5566778", "city": "Faisalabad"},
    {"user_id": 1006, "name": "4 B INDUSTRIES", "cus_id": "99336", "email": "sanaullah@email.com", "phone": "0301-9988776", "city": "Sialkot"},
    {"user_id": 1007, "name": "4W Technologies (Pvt) Ltd", "cus_id": "81033", "email": "zainab.akhtar@email.com", "phone": "0335-4433221", "city": "Quetta"},
    {"user_id": 1008, "name": "A & Z OILS PVT. LTD.", "cus_id": "81626", "email": "imran.siddiqui@email.com", "phone": "0314-7778889", "city": "Multan"},
    {"user_id": 1009, "name": "Hina Sheikh", "cus_id": "82620", "email": "hina.sheikh@email.com", "phone": "0322-6543210", "city": "Sialkot"},
    {"user_id": 1010, "name": "A A BROTHER", "cus_id": "88737", "email": "omar.farooq@email.com", "phone": "0305-1357924", "city": "Sialkot"}
    ]
    return DATA
    
    
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

@function_tool
def fetch_customer_data_tool() -> list:
    """
    Fetch all customer data from the API and return as a list of dicts.
    Use this before applying any filters or counts.
    """
    return fetch_customer_data()

@function_tool
def count_customers(query: str, records: list) -> dict:
    """
    Count customers in the given records.
    If region is mentioned in query, filter by that region first.
    Returns {"total": X, "region": "..."} 
    """
    region = None

    # Region detection from query
    m_region = re.search(r"\bregion\b\s*[:\-]?\s*([A-Za-z0-9\-\(\)\s]+)", query, flags=re.I)
    if not m_region:
        # try without the word 'region' e.g. "Karachi customers"
        m_region = re.search(r"\b([A-Za-z]+)\s+customers?\b", query, flags=re.I)

    if m_region:
        region = m_region.group(1).strip()

    filtered = records
    if region:
        filtered = [r for r in records if region.lower() in str(r.get("region","")).lower()]

    return {
        "total": len(filtered),
        "region": region if region else "all"
    }

@function_tool
def filter_customer_records(
    query: str,
    records: list,
    fields: list[str] = ("cus_id","name","region","phone")
) -> list:
    """Filter customer records based on query (cus_id, name, region)"""
    # Use the manual function to avoid code duplication
    return filter_customers_manual(query, records, fields)
    
customer_agent: Agent = Agent(
    name="Customer Agent",
    instructions=(
        "You are a customer-data specialist. "
        "If the user asks any query, first call `fetch_customer_data_tool` to get records. "
        "Then, if the query is about filtering customers, call `filter_customer_records`. "
        "If the query is about counting customers, call `count_customers`. "
        "Always present the result in a clear way."
    ),
    model=llm_model,
    tools=[fetch_customer_data_tool, filter_customer_records, count_customers],
)


async def main():
    r1 = await Runner.run(customer_agent,'Please display on sialkot region customers in tabular format.')
    print("---->", r1.final_output, "\n")

if __name__ == "__main__":
    asyncio.run(main())