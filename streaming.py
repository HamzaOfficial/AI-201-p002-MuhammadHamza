# def main():
#     print("Hello from agenticfirstproject!")



from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled,function_tool,ModelSettings,handoff,ItemHelpers
import asyncio
import os
from dotenv import load_dotenv,find_dotenv
# from tavily import TavilyClient
from tavily import AsyncTavilyClient
from openai.types.responses import ResponseTextDeltaEvent

load_dotenv(find_dotenv())

gemini_api_key: str | None = os.environ.get("GEMINI_API_KEY")
tavily_api_key: str | None = os.environ.get("TAVILY_API_KEY")
tavily_client = AsyncTavilyClient(api_key=tavily_api_key)
# tavily_client = AsyncTavilyClient(api_key=tavily_api_key)

# Tracing disabled
set_tracing_disabled(disabled=True)

# 1. Which LLM Service?
external_client: AsyncOpenAI = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# 2. Which LLM Model?
llm_model: OpenAIChatCompletionsModel = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=external_client
)

@function_tool
async def search(query:str) -> str:
    """Smart and effecient searching function"""
    print("Searching for query ->",query)
    response = await tavily_client.search(query)
    # print('RESPONSE -->',response)
    return response

@function_tool
async def extract_content(urls: list) -> dict:
    """An extracting urls function"""
    # print('URLS -->',urls)
    response = tavily_client.extract(urls)
    return response

# agent: Agent = Agent(name="Search agent",
#                      model=llm_model,
#                      tools=[search,extract_content],
#                      instructions="You are deep search agent. Use the tools provided to answer. Always use link reference.",
#                      model_settings=ModelSettings(temperature=1.2)) # gemini-2.5 as agent brain - chat completions

# ----------- AGENTS ----------- #
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
    # instructions="You are a research assistant. Write concise, clear, and well-structured summaries.",
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


# ----------- MASTER AGENT (Pipeline with Handoffs) ----------- #
async def main():
    print("\nCALLING AGENT\n")
    # ----------- RUNNER ----------- #
    result = Runner.run_streamed(
        research_team,
        "What are pros and cons of electric cars?"
    )
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            print(event.data.delta, end="", flush=True)
    # async for event in result.stream_events():
    #     print('*'*22)
    #     print('Event-->',event)
        # if event.item.type == "message_output_item":
        #     print(ItemHelpers.text_message_output(event.item))

    #"""Procedure 01"""
    # result: Runner = Runner.run_sync(agent, "What is renewable energy?")
    print("\nCOMPLETING AGENT\n")
    # print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())