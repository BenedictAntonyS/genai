import os
import requests
import yfinance as yf
import streamlit as st

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import Tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

GEMINI_API_KEY = "AIzaSyB6_YiA2NLruc1qtuGrxQc5l_g8Lx5lZmM"
EXCHANGE_API_KEY = "c10a75b5a37ca95e71455efb"

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY
)

# -------- Currency Tool --------
def get_currency(country):
    mapping = {
        "Japan":"JPY",
        "India":"INR",
        "USA":"USD",
        "UK":"GBP",
        "China":"CNY",
        "South Korea":"KRW"
    }
    return mapping.get(country,"Not found")

currency_tool = Tool(
    name="Currency Tool",
    func=get_currency,
    description="Returns official currency of country"
)

# -------- Exchange Rate Tool --------
def get_rates(currency):
    url=f"https://v6.exchangerate-api.com/v6/{EXCHANGE_API_KEY}/latest/{currency}"
    r=requests.get(url).json()
    rates=r["conversion_rates"]
    return f"""
USD:{rates["USD"]}
INR:{rates["INR"]}
GBP:{rates["GBP"]}
EUR:{rates["EUR"]}
"""

rate_tool=Tool(
    name="Rate Tool",
    func=get_rates,
    description="Convert 1 unit currency to USD INR GBP EUR"
)

# -------- Index Tool --------
def get_index(country):
    mapping={
        "Japan":"^N225",
        "India":"^BSESN",
        "USA":"^GSPC",
        "UK":"^FTSE",
        "China":"000001.SS",
        "South Korea":"^KS11"
    }
    ticker=mapping.get(country)
    data=yf.Ticker(ticker)
    price=data.history(period="1d")["Close"].iloc[-1]
    return f"{ticker} index value is {price}"

index_tool=Tool(
    name="Index Tool",
    func=get_index,
    description="Get stock index value"
)

# -------- Maps Tool --------
def get_maps(country):
    mapping={
        "Japan":"https://maps.google.com/?q=Tokyo+Stock+Exchange",
        "India":"https://maps.google.com/?q=BSE+Mumbai",
        "USA":"https://maps.google.com/?q=NYSE",
        "UK":"https://maps.google.com/?q=London+Stock+Exchange",
        "China":"https://maps.google.com/?q=Shanghai+Stock+Exchange",
        "South Korea":"https://maps.google.com/?q=Korea+Exchange"
    }
    return mapping.get(country)

map_tool=Tool(
    name="Maps Tool",
    func=get_maps,
    description="Returns Google maps pin of stock exchange HQ"
)

tools=[currency_tool,rate_tool,index_tool,map_tool]

prompt=hub.pull("hwchase17/react")

agent=create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

executor=AgentExecutor(agent=agent,tools=tools,verbose=True)

st.title("Financial Info LLM Agent")

country=st.text_input("Enter Country")

if st.button("Get Info"):
    res=executor.invoke({
        "input":f"""
Get official currency of {country}.
Convert 1 unit to USD INR GBP EUR.
Get major stock index value.
Provide Google Maps pin of stock exchange HQ.
"""
    })
    st.write(res["output"])
