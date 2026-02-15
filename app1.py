import os
import requests
import yfinance as yf
import streamlit as st

from langchain_groq import ChatGroq
from langchain_core.tools import Tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

# ----------- API KEYS FROM STREAMLIT SECRETS -----------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
EXCHANGE_API_KEY = os.getenv("EXCHANGE_API_KEY")

# ----------- LLM (GROQ - LLAMA3) -----------
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama3-70b-8192"
)

# ----------- CURRENCY TOOL -----------
def get_currency(country):
    mapping = {
        "Japan": "JPY",
        "India": "INR",
        "USA": "USD",
        "UK": "GBP",
        "China": "CNY",
        "South Korea": "KRW"
    }
    return mapping.get(country, "Not found")

currency_tool = Tool(
    name="Currency Tool",
    func=get_currency,
    description="Returns official currency of a country"
)

# ----------- EXCHANGE RATE TOOL -----------
def get_rates(currency):
    url = f"https://v6.exchangerate-api.com/v6/{EXCHANGE_API_KEY}/latest/{currency}"
    r = requests.get(url).json()
    rates = r["conversion_rates"]
    return f"""
USD: {rates["USD"]}
INR: {rates["INR"]}
GBP: {rates["GBP"]}
EUR: {rates["EUR"]}
"""

rate_tool = Tool(
    name="Rate Tool",
    func=get_rates,
    description="Convert 1 unit of currency to USD, INR, GBP, EUR"
)

# ----------- INDEX TOOL -----------
def get_index(country):
    mapping = {
        "Japan": "^N225",
        "India": "^BSESN",
        "USA": "^GSPC",
        "UK": "^FTSE",
        "China": "000001.SS",
        "South Korea": "^KS11"
    }
    ticker = mapping.get(country)
    if ticker:
        data = yf.Ticker(ticker)
        price = data.history(period="1d")["Close"].iloc[-1]
        return f"{ticker} current index value: {price}"
    return "Index not found"

index_tool = Tool(
    name="Index Tool",
    func=get_index,
    description="Returns major stock index current value"
)

# ----------- MAP TOOL -----------
def get_maps(country):
    mapping = {
        "Japan": "https://www.google.com/maps?q=Tokyo+Stock+Exchange&output=embed",
        "India": "https://www.google.com/maps?q=BSE+Mumbai&output=embed",
        "USA": "https://www.google.com/maps?q=NYSE&output=embed",
        "UK": "https://www.google.com/maps?q=London+Stock+Exchange&output=embed",
        "China": "https://www.google.com/maps?q=Shanghai+Stock+Exchange&output=embed",
        "South Korea": "https://www.google.com/maps?q=Korea+Exchange&output=embed"
    }
    return mapping.get(country)

map_tool = Tool(
    name="Maps Tool",
    func=get_maps,
    description="Returns Google Maps embed link of stock exchange HQ"
)

tools = [currency_tool, rate_tool, index_tool, map_tool]

# ----------- AGENT SETUP -----------
prompt = hub.pull("hwchase17/react")

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

# ----------- STREAMLIT UI -----------
st.title("📊 Financial Information LLM Agent")

country = st.text_input("Enter Country Name:")

if st.button("Get Financial Details"):
    if country:
        result = executor.invoke({
            "input": f"""
Get official currency of {country}.
Convert 1 unit of it to USD, INR, GBP, EUR.
Get major stock index value.
"""
        })

        st.subheader("Agent Output")
        st.write(result["output"])

        map_url = get_maps(country)
        if map_url:
            st.subheader("Stock Exchange HQ Location")
            st.markdown(
                f'<iframe src="{map_url}" width="700" height="450"></iframe>',
                unsafe_allow_html=True
            )
    else:
        st.warning("Please enter a country name.")
