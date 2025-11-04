import streamlit as st
from rag_ocr import chat_with_context, load_faiss_index

st.set_page_config(page_title='RAG Demo', layout='wide')

st.title('RAG Demo')
question = st.text_input('Введите вопрос:')

if st.button('Спросить'):
    with st.spinner('Генерация ответа...'):
        answer = chat_with_context(question)
        st.markdown('### Ответ:')
        st.write(answer)
