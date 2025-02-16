import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
import os
from langchain.embeddings import HuggingFaceEmbeddings  
from langchain.vectorstores import FAISS
import google.generativeai as genai
from htmlTemplate import css, bot_template, user_template


load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY")) 

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or "" 
    return text

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    return text_splitter.split_text(raw_text)

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

def chat_with_gemini(question, chat_history):
    model = genai.GenerativeModel("gemini-pro")
    history = "\n".join([f"User: {msg['content']}" if msg['role'] == 'user' else f"Bot: {msg['content']}" for msg in chat_history])
    prompt = f"{history}\nUser: {question}\nBot:"  
    response = model.generate_content(prompt)
    return response.text

def handle_userinput(user_question):
    if st.session_state.vectorstore is not None:
        retriever = st.session_state.vectorstore.as_retriever()
        docs = retriever.get_relevant_documents(user_question)
        context = "\n".join([doc.page_content for doc in docs])
        full_prompt = f"Context: {context}\n\nQuestion: {user_question}"
        response = chat_with_gemini(full_prompt, st.session_state.chat_history)

        st.session_state.chat_history.append({"role": "user", "content": user_question})
        st.session_state.chat_history.append({"role": "bot", "content": response})

        for msg in st.session_state.chat_history:
            if msg['role'] == 'user':
                st.write(user_template.replace("{{MSG}}", msg['content']), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", msg['content']), unsafe_allow_html=True)
    else:
        st.warning("Please upload and process a PDF first.")

def main():
    st.set_page_config(page_title="Chat with MultiPDFs", page_icon="📚")
    st.write(css, unsafe_allow_html=True)

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("Chat with MultiPDFs 📚")
    user_question = st.text_input("Ask your question about your document")

    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on Process", accept_multiple_files=True)

        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.vectorstore = vectorstore
                    st.success("Processing complete! You can now chat with your PDFs.")
            else:
                st.warning("Please upload at least one PDF.")

if __name__ == "__main__":
    main()