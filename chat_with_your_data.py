import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
from dotenv import load_dotenv

from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent

from src.utils import execute_plt_code
from src.logger.base import BaseLogger
from src.models.llms import load_llm
load_dotenv()

logger = BaseLogger()
MODEL_NAME =  "gemini-1.5-pro"


def process_query(da_agent, query):

    response = da_agent(query)

    action = response["intermediate_steps"][-1][0].tool_input["query"]

    if "plt" in action:
        st.write(response["output"])

        fig = execute_plt_code(action, df=st.session_state.df)
        if fig:
            st.pyplot(fig)

        st.write("**Executed code:**")
        st.code(action)

        to_display_string = response["output"] + "\n" + f"```python\n{action}\n```"
        st.session_state.history.append((query, to_display_string))

    else:
        st.write(response["output"])
        st.session_state.history.append((query, response["output"]))

def display_chat_history():
    st.markdown("## Chat History: ")
    for i, (q,r) in enumerate(st.session_state.history):
        st.markdown(f"**Query: {i+1}:** {q}")
        st.markdown(f"**Response: {i+1}:** {r}")
        st.markdown("---")


def main():
    #Set up streamlit interface
    st.set_page_config(
        page_title="Smart Data Analyis Tool",
        page_icon="📊",
        layout="centered"
    )
    st.header("📊 Smart Data Analyis Tool")
    st.write("### Welcome to our data analysis tool. This tool can assits your daily data analysis tasks.")
    #load llms model
    llm = load_llm(model_name=MODEL_NAME)
    logger.info(f'### Succesfully loaded {MODEL_NAME} !###')
    #upload csv file
    with st.sidebar:
        uploaded_file = st.file_uploader("Upload your csv file here", type="csv")
    #initial chat history
    if "history" not in st.session_state:
        st.session_state.history = []
    #read csv file
    if uploaded_file is not None:
        st.session_state.df =  pd.read_csv(uploaded_file)
        st.write("### Your uploaded Data: ", st.session_state.df.head())
    #create data analyis to query with our data
    da_agent = create_pandas_dataframe_agent(
        llm=llm,
        df=st.session_state.df,
        agent_type="tool-calling",
        allow_dangerous_code=True,
        verbose=True,
        return_intermediate_steps=True,
    )
    logger.info("### Successfully loaded data analysis agent !###")
    #input query and process query
    query= st.text_input("Enter your question: ")
    if st.button("Run query"):
        with st.spinner("Please wait..."):
            process_query(da_agent,query)
    #display chat history
    st.divider()
    display_chat_history()
if __name__== "__main__":
    main()