import streamlit as st
from langchain_core.messages import HumanMessage
from agents.agent import TravelAgent

def initialize_agent(csv_file):
    if "agent" not in st.session_state:
        st.session_state.agent = TravelAgent(csv_file)

def render_custom_css():
    st.markdown(
        '''
        <style>
        .main-title {
            font-size: 2.5em;
            color: #333;
            text-align: center;
            margin-bottom: 0.5em;
            font-weight: bold;
        }
        .sub-title {
            font-size: 1.2em;
            color: #333;
            text-align: left;
            margin-bottom: 0.5em;
        }
        .center-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            width: 100%;
        }
        .query-box {
            width: 80%;
            max-width: 600px;
            margin-top: 0.5em;
            margin-bottom: 1em;
        }
        .query-container {
            width: 80%;
            max-width: 600px;
            margin: 0 auto;
        }
        </style>
        ''', unsafe_allow_html=True)

def render_ui():
    st.title("AI TRAVEL CHATBOT")
    st.markdown('<div class="center-container">', unsafe_allow_html=True)
    st.markdown('<div class="query-container">', unsafe_allow_html=True)
    user_input = st.text_area("Enter your travel-related question:", key="query", height=0)

    if st.button("Get travel recommendation"):
        if user_input:
            process_query(user_input)
        else:
            st.error("Please enter a region")

def process_query(user_input):
    try:
        messages = [HumanMessage(content=user_input)]
        result = st.session_state.agent.graph.invoke({"messages": messages})

        # st.subheader("Recommended travel destinations and descriptions")
        # st.write(result["messages"][-1].content)
        st.write_stream(result["messages"][-1].content)

    except Exception as e:
        st.error(f"Error: {e}")

def main():
    csv_file = "touristAttractions.csv"
    initialize_agent(csv_file)
    render_custom_css()
    render_ui()

if __name__ == "__main__":
    main()
