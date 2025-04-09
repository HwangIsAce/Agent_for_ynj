from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_community.chat_models import ChatOllama
from typing import Annotated, TypedDict
import operator

import streamlit as st

import pandas as pd
import os
from dotenv import load_dotenv

from agents.tools.flights_finder import flights_finder
from agents.tools.hotels_finder import hotels_finder

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

        self.tools = {t.name: t for t in [flights_finder, hotels_finder]}
        self.tools_llm = llm.bind_tools([flights_finder, hotels_finder])

        builder = StateGraph(AgentState)
        builder.add_node("analyze_request", self.analyze_request)
        builder.add_node("search_travel_data", self.search_travel_data)
        builder.add_node("generate_recommendation", self.generate_recommendation)
        builder.add_node("call_tools_llm", self.call_tools_llm)
        builder.add_node("invoke_tools", self.invoke_tools)

        builder.set_entry_point("analyze_request")

        builder.add_conditional_edges(
            "analyze_request",
            self.decide_next_step,
            {"search": "search_travel_data", "tools": "call_tools_llm", "end": END}
        )
        builder.add_edge("search_travel_data", "generate_recommendation")
        builder.add_edge("generate_recommendation", END)
        builder.add_edge("call_tools_llm", "invoke_tools")
        builder.add_edge("invoke_tools", END)

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
    
    def call_tools_llm(self, state: AgentState):
        messages = state["messages"]
        prompt = "당신은 여행 전문가입니다. 필요한 경우 항공권이나 호텔을 조회하는 도구를 사용할 수 있습니다."
        messages = [SystemMessage(content=prompt)] + messages
        message = self.tools_llm.invoke(messages)
        return {"messages": message}
    
    def invoke_tools(self, state: AgentState):
        tool_calls = state["messages"][-1].tool_calls
        results = []
        for t in tool_calls:
            if not t['name'] in self.tools:
                result = "잘못된 도구 이름입니다. 다시 시도하세요."
            else:
                result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        return {"messages": results}

    def decide_next_step(self, state: AgentState):
        query = state["messages"][-1].content
        if "항공" in query or "비행기" in query or "호텔" in query:
            return "tools"
        elif "??" in query or "??" in query:
            return "search"
        return "end"
