import os
import streamlit as st
import requests
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import Tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub


# Read API keys from environment variables for safety
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyDJ22m_KzJ1QXEvxZFeBulR_ogVgNHrgqk")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "484d4c642f6187b98ea80ab3d64e6e51")

# Initialize LLM if key provided
llm = None
if GEMINI_API_KEY:
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=GEMINI_API_KEY
        )
    except Exception:
        llm = None

# ---------------- WEATHER TOOL ----------------
def get_weather(city: str) -> str:
    if not WEATHER_API_KEY:
        return "Weather API key not configured."
    try:
        url = "http://api.openweathermap.org/data/2.5/weather"
        params = {"q": city, "appid": WEATHER_API_KEY, "units": "metric"}
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("main"):
            temp = data["main"]["temp"]
            desc = data["weather"][0]["description"]
            return f"Current temperature in {city} is {temp}°C with {desc}."
        return "Weather data not available."
    except Exception as e:
        return f"Error fetching weather: {e}"

weather_tool = Tool(
    name="Weather Tool",
    func=get_weather,
    description="Gets current weather of a city"
)

# ---------------- FLIGHT TOOL (Mock) ----------------
def get_flights(city: str):
    return f"Sample flight: Chennai to {city} | ₹45,000 | 7h 30m | ANA Airlines"

flight_tool = Tool(name="Flight Tool", func=get_flights, description="Provides flight options")

# ---------------- HOTEL TOOL (Mock) ----------------
def get_hotels(city: str) -> str:
    return f"Sample hotel in {city}: Sakura Grand Hotel | ₹6,000 per night | 4.5 stars"

hotel_tool = Tool(name="Hotel Tool", func=get_hotels, description="Provides hotel options")

tools = [weather_tool, flight_tool, hotel_tool]

agent = None

if llm:
    prompt_template = hub.pull("hwchase17/react")

    react_agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt_template
    )

    agent = AgentExecutor(
        agent=react_agent,
        tools=tools,
        verbose=True
    )

# ---------------- STREAMLIT UI ----------------
st.title("AI Trip Planner Agent 🌍")
st.write("Provide a destination and I'll plan a short trip. (Requires GEMINI_API_KEY and WEATHER_API_KEY)")

user_input = st.text_input("Enter your trip request:")

if st.button("Plan Trip"):
    if not user_input:
        st.error("Please enter a trip request.")
    elif not agent:
        st.error("LLM not configured. Set GEMINI_API_KEY environment variable to enable agent.")
    else:
        prompt = f"""
        Plan the trip and include:
        1. One paragraph about city cultural & historic significance just frame something and give it
        
        2. Current weather
        3. Travel dates (assume May 10-13 if not specified)
        4. Flight options
        5. Hotel options
        6. Detailed itinerary by day

        User request: {user_input}
        """
        try:
           response = agent.invoke({"input": prompt})        
        except Exception as e:
            response = f"Agent error: {e}"
        st.write(response["output"])
