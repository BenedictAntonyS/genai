import os
import requests
import yfinance as yf
import streamlit as st

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import Tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

GEMINI_API_KEY = "YOUR_GEMINI_KEY"
EXCHANGE_API_KEY = "YOUR_EXCHANGE_KEY"

# -------- LLM --------
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
USD:{rates['USD']}
INR:{rates['INR']}
GBP:{rates['GBP']}
EUR:{rates['EUR']}
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

# -------- Maps Tool (EMBED HTML) --------
def get_maps(country):
    mapping={
        "Japan":"Tokyo Stock Exchange",
        "India":"BSE Mumbai",
        "USA":"NYSE",
        "UK":"London Stock Exchange",
        "China":"Shanghai Stock Exchange",
        "South Korea":"Korea Exchange"
    }

    place=mapping.get(country)

    iframe=f"""
    <iframe
        width="600"
        height="450"
        style="border:0"
        loading="lazy"
        allowfullscreen
        src="https://www.google.com/maps?q={place}&output=embed">
    </iframe>
    """

    return iframe

map_tool=Tool(
    name="Maps Tool",
    func=get_maps,
    description="Returns embedded map of stock exchange HQ"
)

tools=[currency_tool,rate_tool,index_tool,map_tool]

# -------- Agent --------
prompt=hub.pull("hwchase17/react")

agent=create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

executor=AgentExecutor(agent=agent,tools=tools,verbose=True)

# -------- UI --------
st.title("Financial Info LLM Agent")

country=st.text_input("Enter Country")

if st.button("Get Info"):
    res=executor.invoke({
        "input":f"""
Get official currency of {country}.
Convert 1 unit to USD INR GBP EUR.
Get major stock index value.
Use Maps Tool to show stock exchange HQ.
"""
    })

    st.write(res["output"])

    # Render map separately
    map_html=get_maps(country)
    st.components.v1.html(map_html,height=450)
