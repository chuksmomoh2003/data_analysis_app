#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import os
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.agents.agent_types import AgentType
import base64

# Initialize session state for history
if 'history' not in st.session_state:
    st.session_state.history = []

# Function to save responses to session state
def save_to_history(query, response):
    st.session_state.history.append({'Query': query, 'Response': response})

# Sidebar for API key input
with st.sidebar:
    api_key = st.text_input('Enter OpenAI API Key:', type='password')
    if api_key:
        os.environ['OPENAI_API_KEY'] = api_key
    else:
        st.error("Please enter the OpenAI API Key to proceed.")

# Check if API key is provided
if api_key:
    # Title of the app
    st.title('Chat With Your Data')
    st.image('data_analysis.jpeg')

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    # Check if a file is uploaded
    if uploaded_file is not None:
        # Read the uploaded file into a DataFrame
        df = pd.read_csv(uploaded_file)

        # Create a Langchain agent
        agent = create_pandas_dataframe_agent(
        ChatOpenAI(temperature=0, model="gpt-3.5-turbo"), df, verbose=True, agent_type=AgentType.OPENAI_FUNCTIONS,
        handle_parsing_errors=True
        )

        # Query input
        query = st.text_input("Enter your query")

        # Process query
        if st.button('Submit Query'):
            response = agent.run(query)
            save_to_history(query, response)
            st.write("Response:", response)

        # Display previous queries and responses
        st.subheader("Previous Queries and Responses")
        for item in st.session_state.history:
            st.text(f"Q: {item['Query']}\nA: {item['Response']}")

        # Download button for the history
        if st.button('Download Responses'):
            history_df = pd.DataFrame(st.session_state.history)
            csv = history_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
            href = f'<a href="data:file/csv;base64,{b64}" download="query_history.csv">Download csv file</a>'
            st.markdown(href, unsafe_allow_html=True)

    else:
        st.write("Please upload a CSV file to begin.")


# In[ ]:




