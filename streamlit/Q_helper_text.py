import streamlit as st
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.output_parsers.openai_tools import JsonOutputKeyToolsParser
from typing import List
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
    RunnableAssign
)
import os

from langsmith import traceable


os.environ["LANGCHAIN_API_KEY"]=st.secrets['LANGCHAIN_API_KEY']
os.environ["LANGCHAIN_TRACING_V2"]=st.secrets['LANGCHAIN_TRACING_V2']
os.environ["LANGCHAIN_ENDPOINT"]=st.secrets['LANGCHAIN_ENDPOINT']
os.environ['LANGCHAIN_PROJECT']=st.secrets['LANGCHAIN_PROJECT']




#openai_api_key = os.environ.get('OPENAI_API_KEY')

### chroma는 semantic chunking, chroma_recursivesplit은 recursivecharactersplit으로 chunking한 것입니다.



@traceable
def refine_question(model,question):
    prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a Korean doctor consulting a patient. When given a medical complaint, ask a list of questions that would help you with the consultation."""),
    ("user", """
{question}""")
    
])



    llm = ChatOpenAI(model_name=model, temperature = 0)  #gpt-4-turbo-preview #gpt-3.5-turbo #gpt-4 #gpt-3.5-turbo-1106
    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser

    opinion = chain.invoke(question)        
    return opinion
    

    
if "opinion" not in st.session_state:
    st.session_state.opinion = ''
if "disabled" not in st.session_state:
    st.session_state.disabled = True
    
    
st.title('질문작성 도우미')


def add_opinion():
    opinion = refine_question(model="gpt-4o",question = st.session_state.text_area)
    st.session_state.text_area = st.session_state.text_area +'\n\n'+ opinion

openai_api_key = st.sidebar.text_input('OpenAI API Key', value = '',type='password')
with st.sidebar:
    st.session_state
    
if openai_api_key =='medihub':
    openai_api_key = st.secrets['OPENAI_API_KEY']
    
if openai_api_key.startswith('sk-'):
    st.session_state.disabled = False

with st.form('my_form'):
    st.text_area('질문 작성 도우미는 한번에 원하시는 답변을 받으실 수 있도록, 사용자님의 질문을 검토해드립니다.', placeholder='언제부터, 어디가, 어떻게 불편하신지 등 증상에 대해 가능한 자세히 작성해질수록 더 자세한 답변을 드릴 수 있습니다.',key='text_area', height=400)
    submitted = st.form_submit_button('질문 작성 도우미',on_click=add_opinion,disabled=st.session_state.disabled)
    if not openai_api_key.startswith('sk-'):
        st.warning('Please enter your OpenAI API key!', icon='⚠')
        
