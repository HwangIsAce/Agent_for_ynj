from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langchain_community.chat_models import ChatOllama
from typing import Annotated, TypedDict
import operator

import streamlit as st

import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

def load_travel_data(file_path):
    try:
        df = pd.read_csv(file_path, encoding='utf-8') 
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='cp949') 
    return df

def find_travel_destinations(region, df):
    filtered_df = df[df['??'].str.contains(region, na=False, case=False)]  
    if filtered_df.empty:
        return None
    return filtered_df[['??', '??', '??']].to_dict(orient='records')  

llm = ChatOllama(model="timHan/llama3korean8B4QKM:latest", temperature=0, stream=True)

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

TRAVEL_SYSTEM_PROMPT = """
??? ?? ?? ??????.
- ???? ???? ???? ?????.
- ?? ?? ???? ??? ?, ???? ???? CSV ???? ???? ?????.
- ??? ???? ????, ? ??? ???? ?? ??? ?????.
"""

class TravelAgent:
    def __init__(self, csv_file):
        self.travel_data = load_travel_data(csv_file)

        builder = StateGraph(AgentState)
        builder.add_node("analyze_request", self.analyze_request)
        builder.add_node("search_travel_data", self.search_travel_data)
        builder.add_node("generate_recommendation", self.generate_recommendation)

        builder.set_entry_point("analyze_request")

        builder.add_conditional_edges(
            "analyze_request",
            self.decide_next_step,
            {"search": "search_travel_data", "end": END}
        )
        builder.add_edge("search_travel_data", "generate_recommendation")
        builder.add_edge("generate_recommendation", END)

        self.graph = builder.compile()

    def analyze_request(self, state: AgentState):
        messages = state["messages"]
        query = messages[-1].content
        system_message = SystemMessage(content=TRAVEL_SYSTEM_PROMPT)
        
        response = llm.stream([system_message, HumanMessage(content=query)])
        
        extracted_location = ""
        
        response_container = st.empty()

        for chunk in response:
            if hasattr(chunk, "content"):
                extracted_location += chunk.content
                response_container.write(extracted_location)

        extracted_location = response

        return {"messages": messages + [HumanMessage(content=extracted_location)]}

    def search_travel_data(self, state: AgentState):
        messages = state["messages"]
        region = messages[-1].content

        travel_recommendations = find_travel_destinations(region, self.travel_data)

        if travel_recommendations:
            return {"messages": messages + [HumanMessage(content=str(travel_recommendations))]}
        else:
            return {"messages": messages + [HumanMessage(content=f"{region}?? ??? ???? ?? ? ????.")]}

    def generate_recommendation(self, state: AgentState):
        messages = state["messages"]
        travel_data = messages[-1].content

        recommendation_prompt = f"""
        ?? ??? ??? ????, ?? ??? ???? ?? ??? ?????.
        {travel_data}
        """
        
        response = llm.invoke([SystemMessage(content=recommendation_prompt)])
        return {"messages": messages + [HumanMessage(content=response.content)]}

    def decide_next_step(self, state: AgentState):
        query = state["messages"][-1].content
        if "??" in query or "??" in query:
            return "search"
        return "end"
