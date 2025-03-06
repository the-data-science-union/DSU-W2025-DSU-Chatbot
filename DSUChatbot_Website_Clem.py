import requests
from bs4 import BeautifulSoup
import json
url = “https://datascienceunion.com/team”
response = requests.get(url)
soup = BeautifulSoup(response.content, ‘html.parser’)
name_elements = soup.find_all(‘h2’)
info_elements = soup.find_all(‘p’)
names = [
    name.text.strip()
    for name in name_elements
    if “Executive Board” not in name.text and “Members” not in name.text
]
members_info = []
for info in info_elements:
    text = info.text.strip()
    if “Class of” in text:
        members_info.append({“type”: “year”, “value”: text})
    else:
        members_info.append({“type”: “title”, “value”: text})
members = {}
for i, name in enumerate(names):
    if i < len(members_info):
        info = members_info[i]
        if info[“type”] == “year”:
            members[name] = {“title”: “Member”, “year”: info[“value”]}
        else:
            members[name] = {“title”: info[“value”], “year”: “Unknown Year”}
    else:
        members[name] = {“title”: “Member”, “year”: “Unknown Year”}
with open(“members.json”, “w”) as file:
    json.dump(members, file, indent=4)
print(“Scraped data saved to members.json!“)
12:47
Data Loader:
12:48
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
# Define chunk parameters
CHUNK_SIZE = 400
CHUNK_OVERLAP = 100
# Load JSON data
with open(“members.json”, “r”) as file:
    data = json.load(file)
# Format and text, ensure missing data is handled
formatted_text = “\n”.join([
    f”{name}: {info[‘title’]}, {info[‘year’]}, {info.get(‘Description’, ‘No description available’)}”
    for name, info in data.items()
])
# Create RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)
# Generate text chunks
chunks = splitter.split_text(formatted_text)
chunks_dict = {f”chunk_{i+1}“: chunk for i, chunk in enumerate(chunks)}
with open(“chunks.json”, “w”) as file:
    json.dump(chunks_dict, file, indent=4)
12:49
Vector Embeddings
12:49
from sentence_transformers import SentenceTransformer
import json
model = SentenceTransformer(‘all-MiniLM-L6-v2’)
# Load the chunks from the JSON file
with open(“chunks.json”, “r”) as file:
    chunks_dict = json.load(file)
# Extract chunk texts
chunks_text = list(chunks_dict.values())
# Generate embeddings for the chunks
embeddings = model.encode(chunks_text)
# Store embeddings alongside the chunk info
chunks_with_embeddings = {
    chunk_id: {
        “text”: chunk_text,
        “embedding”: embedding.tolist()
    }
    for chunk_id, (chunk_text, embedding) in zip(chunks_dict.keys(), zip(chunks_text, embeddings))
}
# Save the chunks and embeddings to a new JSON file
with open(“chunks_with_embeddings.json”, “w”) as file:
    json.dump(chunks_with_embeddings, file, indent=4)
print(“Embeddings for chunks have been saved to chunks_with_embeddings.json.“)
12:49
Vector Store:
12:49
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import os
import json
# Load the JSON file with your chunks
with open(‘chunks_with_embeddings.json.json’, ‘r’) as file:
    chunk_data = json.load(file)
chunks = [entry[‘text’] for entry in chunk_data]
embedding_model = HuggingFaceEmbeddings(model_name=“BAAI/bge-large-en”)
persist_directory = ‘/Users/clem/chatbot/DSU_LangChain_InternalRAG’
# Clean up old database files if any
if os.path.exists(persist_directory):
    !rm -rf {persist_directory}  # Remove old database files
# Create the vector store using Chroma
vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=persist_directory
)
# Check the number of vectors stored in the vector store
print(vectordb._collection.count())
# Check the length of the chunks to ensure they’re correctly processed
print(len(chunks))
12:50
Similarity Search / LLM Wrapper
12:51
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
12:51
with open(‘chunks_with_embeddings.json’, ‘r’) as file:
    chunks_dict = json.load(file)
12:51
chunk_ids = list(chunks_dict.keys())
texts = [chunks_dict[chunk_id][“text”] for chunk_id in chunk_ids]
embeddings = np.array([chunks_dict[chunk_id][“embedding”] for chunk_id in chunk_ids])
# Load the same model used for embedding generation
model = SentenceTransformer(‘all-MiniLM-L6-v2’)
def similarity_search(query_text, top_k=3):
    “”"Finds the most similar chunks to a given query text.“”"
    # Generate the query embedding
    query_embedding = model.encode([query_text])[0]  # Convert to NumPy array
    # Compute cosine similarity
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    # Get top_k indices sorted by highest similarity
    top_indices = np.argsort(similarities)[::-1][:top_k]
    # Return the most similar chunks
    return [(chunk_ids[i], texts[i], similarities[i]) for i in top_indices]
# Example query
query_text = “Explain machine learning models”  # Replace with your own query
# Perform similarity search
results = similarity_search(query_text, top_k=5)
# Print the results
for chunk_id, text, score in results:
    print(f”Chunk ID: {chunk_id} | Similarity: {score:.4f}\nText: {text}\n”)
12:51
import os
import openai
12:51
openai.api_key = ‘REDACTED’
12:52
import json
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
persist_directory = “Desktop/chatbot/DSU_LangChain_InternalRAG/chroma_db”
with open(“chunks_with_embeddings.json”, “r”) as f:
    chunks_dict = json.load(f)
texts = [chunks_dict[chunk_id][“text”] for chunk_id in chunks_dict]
embeddings = [chunks_dict[chunk_id][“embedding”] for chunk_id in chunks_dict]
embedding_model = OpenAIEmbeddings(openai_api_key=openai.api_key)  # Replace with your key
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
vectordb.add_texts(texts=texts, embeddings=embeddings)
vectordb.persist()
print(“Embeddings successfully stored in ChromaDB.“)
12:52
from langchain.prompts import PromptTemplate
# Build prompt
template = “”"Use the following pieces of context to answer the question at the end. If you don’t know the answer, just say that you don’t know, don’t try to make up an answer.
Use three sentences maximum. Keep the answer as concise as possible.
Always say “thanks for asking!” at the end of the answer.
Context: {context}
Question: {question}“”"
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
12:52
# Run chain
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={“prompt”: QA_CHAIN_PROMPT}
)
12:52
question = “What does Charlie Hoose do?”
result = qa_chain({“query”: question})
result[“result”]