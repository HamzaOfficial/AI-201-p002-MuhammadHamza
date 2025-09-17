import os
import asyncio
import streamlit as st
import matplotlib.pyplot as plt
import re
from dotenv import load_dotenv
from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, set_tracing_disabled, function_tool
import tempfile
import pandas as pd
from PIL import Image

# Load environment variables
load_dotenv()
set_tracing_disabled(disabled=True)

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

# Setup Gemini client
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

external_client = AsyncOpenAI(api_key=GEMINI_API_KEY, base_url=BASE_URL)
llm_model = OpenAIChatCompletionsModel(model="gemini-2.5-flash", openai_client=external_client)

def fetch_customer_data(filters=None):
    df = pd.read_csv("customers_data.csv")
    DATA = [
    {"user_id": 1001, "name": "ZUBAIR HASSAN (B12,COLONY3 CHAKLALA GARRISON RAWALPINDI)", "cus_id": "83907", "email": "ali.ahmed@email.com", "phone": "0300-1234567", "region": "Lahore"},
    {"user_id": 1002, "name": "(FWO) PD HQ NBBIAP", "cus_id": "88594", "email": "fatima.khan@email.com", "phone": "0312-9876543", "region": "Karachi"},
    {"user_id": 1003, "name": "156 INDEPENDENT INFANTRY WORKSHOP COMPANY", "cus_id": "81155", "email": "usman.malik@email.com", "phone": "0333-4567890", "region": "Islamabad"},
    {"user_id": 1004, "name": "1st for Connect Pvt Ltd", "cus_id": "83025", "email": "ayesha.raza@email.com", "phone": "0345-1122334", "region": "Karachi"},
    {"user_id": 1005, "name": "3G TECHNOLOGIES (GURANTEE) LIMITED", "cus_id": "82989", "email": "bilal.hassan@email.com", "phone": "0321-5566778", "region": "Multan"},
    {"user_id": 1006, "name": "4 B INDUSTRIES", "cus_id": "99336", "email": "sanaullah@email.com", "phone": "0301-9988776", "region": "Karachi"},
    {"user_id": 1007, "name": "4W Technologies (Pvt) Ltd", "cus_id": "81033", "email": "zainab.akhtar@email.com", "phone": "0335-4433221", "region": "Islamabad"},
    {"user_id": 1008, "name": "A & Z OILS PVT. LTD.", "cus_id": "81626", "email": "imran.siddiqui@email.com", "phone": "0314-7778889", "region": "Multan"},
    {"user_id": 1009, "name": "Hina Sheikh", "cus_id": "82620", "email": "hina.sheikh@email.com", "phone": "0322-6543210", "region": "Lahore"},
    {"user_id": 1010, "name": "A A BROTHER", "cus_id": "88737", "email": "omar.farooq@email.com", "phone": "0305-1357924", "region": "Lahore"}
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
    """Filter customer records based on query (cus_id, name, region,email,phone,mobile)"""
    # Use the manual function to avoid code duplication
    return filter_customers_manual(query, records, fields)

@function_tool
def visualize_customer_data(query: str, records: list) -> str:
    """
    Dynamically generate plots from customer data based on the query.
    Automatically detects which columns to visualize based on query context.
    Supports bar charts, pie charts, and other visualization types.
    Returns the file path of the saved plot.
    """
    if not records:
        return "No records available for plotting."
    
    # Available columns in the data
    available_columns = list(records[0].keys())
    
    # Analyze query to determine what to visualize
    query_lower = query.lower()
    
    # Detect visualization type from query
    if "pie" in query_lower or "distribution" in query_lower:
        chart_type = "pie"
    elif "line" in query_lower or "trend" in query_lower:
        chart_type = "line"
    elif "scatter" in query_lower or "correlation" in query_lower:
        chart_type = "scatter"
    else:
        chart_type = "bar"  # default
    
    # Detect which column to visualize based on query context
    column_to_visualize = None
    
    # Check for explicit column mentions in query
    for column in available_columns:
        if column in query_lower:
            column_to_visualize = column
            break
    
    # If no explicit column mentioned, use intelligent detection
    if not column_to_visualize:
        if "region" in available_columns and ("region" in query_lower or "location" in query_lower or "region" in query_lower):
            column_to_visualize = "region"
        elif "name" in available_columns and ("name" in query_lower or "customer" in query_lower):
            column_to_visualize = "name"
        elif "id" in available_columns and ("id" in query_lower or "number" in query_lower):
            column_to_visualize = "cus_id"
        else:
            # Default to first categorical column
            categorical_columns = [col for col in available_columns 
                                 if isinstance(records[0].get(col), (str, int)) and col != "user_id"]
            column_to_visualize = categorical_columns[0] if categorical_columns else available_columns[0]
    
    # Count values for the selected column
    value_counts = {}
    for record in records:
        value = str(record.get(column_to_visualize, "Unknown"))
        value_counts[value] = value_counts.get(value, 0) + 1
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    
    if chart_type == "pie":
        plt.pie(value_counts.values(), labels=value_counts.keys(), autopct='%1.1f%%')
        plt.title(f"Distribution of Customers by {column_to_visualize.capitalize()}")
    elif chart_type == "bar":
        plt.bar(value_counts.keys(), value_counts.values())
        plt.title(f"Customers by {column_to_visualize.capitalize()}")
        plt.xlabel(column_to_visualize.capitalize())
        plt.ylabel("Count")
        plt.xticks(rotation=45)
    else:
        # Default to bar chart
        plt.bar(value_counts.keys(), value_counts.values())
        plt.title(f"Customers by {column_to_visualize.capitalize()}")
        plt.xlabel(column_to_visualize.capitalize())
        plt.ylabel("Count")
        plt.xticks(rotation=45)
    
    # Save file with timestamp to avoid overwriting
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = f"customer_plot_{column_to_visualize}_{timestamp}.png"
    
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()
    print(file_path)
    
    return file_path

# Create the customer agent
customer_agent = Agent(
    name="Customer Agent",
    instructions=(
        "You are a customer-data specialist. "
        "If the user asks any query, first call `fetch_customer_data_tool` to get records. "
        "Then analyze what the user wants:\n"
        "1. If they want filtered data, call `filter_customer_records`\n"
        "2. If they want counts, call `count_customers`\n"
        "3. If they mention graph, chart, plot, visualization, or any visual representation, "
        "call `visualize_customer_data` with the user's query and the records\n"
        "visualize_customer_data will always return the graph path print it in the display"
        "Always present the result in a clear way."
    ),
    model=llm_model,
    tools=[fetch_customer_data_tool, filter_customer_records, count_customers, visualize_customer_data],
)

# Streamlit UI
st.set_page_config(page_title="Customer Data Agent", page_icon="📊", layout="wide")

st.title("📊 Customer Data Visualization Agent")
st.markdown("Ask questions about customer data and get visualizations!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
# if prompt := st.chat_input("Ask about customer data (e.g., 'Show me customers from Sialkot', 'Create a bar chart of customers by city')"):
#     # Add user message to chat history
#     st.session_state.messages.append({"role": "user", "content": prompt})
    
#     # Display user message in chat message container
#     with st.chat_message("user"):
#         st.markdown(prompt)
    
#     # Display assistant response in chat message container
#     with st.chat_message("assistant"):
#         message_placeholder = st.empty()
#         message_placeholder.markdown("Thinking...")
        
#         # Run the agent
#         try:
#             result = asyncio.run(Runner.run(customer_agent, prompt))
#             print('Result1 -- > ',result)
#             print('Result2 -- > ',result.final_output)
            
#             # Check if the result contains a plot file path
#             if isinstance(result.final_output, str) and result.final_output.endswith('.png'):
#                 # Display the image
#                 try:
#                     image = Image.open(result.final_output)
#                     st.image(image, caption="Generated Visualization")
#                     response = f"Here's your visualization: {result.final_output}"
#                 except Exception as e:
#                     response = f"Error displaying image: {str(e)}. File path: {result.final_output}"
#             else:
#                 response = result.final_output
                
#             message_placeholder.markdown(response)
#             st.session_state.messages.append({"role": "assistant", "content": response})
            
#         except Exception as e:
#             error_msg = f"Error: {str(e)}"
#             message_placeholder.markdown(error_msg)
#             st.session_state.messages.append({"role": "assistant", "content": error_msg})


# Accept user input
if prompt := st.chat_input("Ask about customer data (e.g., 'Show me customers from Sialkot', 'Create a bar chart of customers by region')"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt, "type": "text"})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        
        # Run the agent
        try:
            result = asyncio.run(Runner.run(customer_agent, prompt))
            
            # Parse the response to check if it contains an image path
            response_text = result.final_output
            
            # Check if the response contains a PNG file path (even with descriptive text)
            png_pattern = r'(\S+\.png)'
            png_match = re.search(png_pattern, response_text)
            
            if png_match:
                image_path = png_match.group(1)
                
                # Display the message text
                message_text = response_text.replace(image_path, "").strip()
                if message_text:
                    st.markdown(message_text)
                
                # Display the image if it exists
                if os.path.exists(image_path):
                    image = Image.open(image_path)
                    st.image(image, caption="Generated Visualization")
                    
                    # Add to chat history
                    if message_text:
                        st.session_state.messages.append({"role": "assistant", "content": message_text, "type": "text"})
                    st.session_state.messages.append({"role": "assistant", "content": image_path, "type": "image"})
                else:
                    st.error(f"Image file not found: {image_path}")
                    st.session_state.messages.append({"role": "assistant", "content": f"{response_text}\n\nError: Image file not found: {image_path}", "type": "text"})
            else:
                # Regular text response
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text, "type": "text"})
                
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg, "type": "text"})


# Sidebar with information and examples
with st.sidebar:
    st.header("About")
    st.markdown("""
    This AI agent helps you analyze and visualize customer data.
    
    **Available Functions:**
    - Fetch customer data
    - Filter customers by various criteria
    - Count customers by region/city
    - Create visualizations (bar charts, pie charts, etc.)
    """)
    
    st.header("Example Queries")
    st.markdown("""
    - "Show me customers from Sialkot"
    - "How many customers are in Karachi?"
    - "Create a bar chart of customers by region"
    - "Show me a pie chart of customer distribution"
    - "Filter customers with ID 83907"
    """)
    
    # Display raw data if requested
    if st.checkbox("Show raw data"):
        st.write(fetch_customer_data())

# Add some styling
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stChatInput {
        position: fixed;
        bottom: 3rem;
    }
</style>
""", unsafe_allow_html=True)
