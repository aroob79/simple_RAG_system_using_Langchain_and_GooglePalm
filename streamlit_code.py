import streamlit as st
from keys import api_keys
from langchain_code import llm_loading, llm_QA

st.title('QA System')

st.sidebar.title('Input URLs')
urls = []
for i in range(3):
    url = st.sidebar.text_input(f'URL {i+1}')
    if url:
        urls.append(url)

press = st.sidebar.button('Press')
main_text = st.empty()

if press:
    st.session_state.chai_ = llm_loading(api_keys, urls, main_text)

if "chai_" in st.session_state:
    query = main_text.text_input("Question:")
    if query:
        res, source = llm_QA(query, st.session_state.chai_)
        st.header("Answer")
        st.subheader(res)
        st.subheader("Source:")
        st.write(source)
