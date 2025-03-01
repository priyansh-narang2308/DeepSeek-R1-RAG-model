import os

import streamlit as st

from rag_utility import process_document_to_chroma_db,answer_question

# set the working directory
working_dir = os.getcwd()

st.title("🐋 DeepSeek-R1 - Document RAG")

# file uploader widget
uploaded_file = st.file_uploader("Upload a PDF file!", type=["pdf"])

if uploaded_file is not None:
    # define the save path
    save_path = os.path.join(working_dir, uploaded_file.name)
    # save the file now,write it to save the file
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    process_document=process_document_to_chroma_db(uploaded_file.name)
    # this shows the info written
    st.info("Document Processed Successfully! ")

# text-widget to get user questions
user_question = st.text_area("Ask Your Question about the document..")

if st.button("Answer"):
    answer=answer_question(user_question)
    st.markdown("### DeepSeek-R1 Response")
    st.markdown(answer)


