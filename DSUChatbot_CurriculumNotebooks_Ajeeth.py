import os
import re
import json
import glob
import openai
import textwrap
from PIL import Image
import streamlit as st
from typing import List, Dict
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Streamlit UI Setup
st.title("DSU Curriculum Chatbot")
st.write("Ask me anything related to the DSU curriculum notebooks!")

# Load and process curriculum notebooks
def load_ipynb(file_path: str) -> List[Dict]:
    with open(file_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    return notebook.get('cells', [])

def clean_text(text: str) -> str:
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'\S+\.(gif|png|jpg|jpeg|svg|bmp|tiff)', '', text, flags=re.IGNORECASE)
    return text.strip()

def extract_text(cells: List[Dict]) -> List[str]:
    texts = []
    for cell in cells:
        if cell['cell_type'] == 'markdown':
            cell_text = "\n".join(cell['source'])
            cleaned_text = clean_text(cell_text)
            if cleaned_text:
                texts.append(cleaned_text)
        elif cell['cell_type'] == 'code':
            code_text = "\n".join(cell['source'])
            cleaned_text = clean_text(code_text)
            if cleaned_text:
                texts.append(f"```python\n{cleaned_text}\n```")
    return texts

def split_into_chunks(texts: List[str], chunk_size: int = 512) -> List[str]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=50, separators=["\n\n", "\n", " "])
    return text_splitter.split_text("\n\n".join(texts))

# Load vector database only once
if "vectordb" not in st.session_state:
    folderpath = "Curriculum Notebooks Copy (DSU Chatbot Project)"
    os.chdir("/Users/ajeethiyer/Documents/Extracurriculars/DSU/LangChain")
    files = glob.glob(f"{folderpath}/*.ipynb")

    files_and_chunks = {}
    for file in files:
        cells = load_ipynb(file)
        text_content = extract_text(cells)
        chunks = split_into_chunks(text_content)
        files_and_chunks[file] = chunks

    documents = []
    for file, chunks in files_and_chunks.items():
        for idx, chunk in enumerate(chunks):
            doc = Document(page_content=chunk, metadata={"file_name": file, "chunk_id": idx})
            documents.append(doc)

    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en")
    persist_directory = 'curriculum_vector_store'

    st.session_state.vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_directory
    )

openai.api_key = 'REDACTED'

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai.api_key)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

QA_CHAIN_PROMPT = PromptTemplate.from_template(
    """Use the following pieces of context to answer the question at the end. 
Use five sentences maximum. Keep the answer concise. If you can answer the question based on the database, please answer it.
Say "thanks for asking!" if you can answer the question at the end of your answer. To be clear, contributors are connected to the DSU curriculum notebooks that they made. DSU stands for Data Science Union.

    {chat_history}  
    Context: {context}  
    Question: {question} """
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=st.session_state.vectordb.as_retriever(),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT}
)

# Streamlit Chat Interface
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Track the search input field using session state
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# Text input with key to control the state
user_input = st.text_input("Enter your question:", value=st.session_state.user_input)

if user_input:
    # Process the query
    result = qa_chain({"question": user_input})
    st.session_state.chat_history.append((user_input, result['answer']))

    # Clear the input field after processing
    st.session_state.user_input = ""  # Reset the input field

# Display chat history
for question, answer in st.session_state.chat_history:
    st.write(f"**Q:** {question}")
    st.write(f"**A:** {textwrap.fill(answer, width=120)}")




# Load the image
image = Image.open("dsu_logo.png")

# Display the image in your Streamlit app with an optional caption
st.image(image, caption="", use_column_width=True)

