import os
import re
import json
import dotenv
import tiktoken
import streamlit as st
from html import unescape
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.vectorstores.faiss import FAISS
from langchain.docstore.document import Document
from langchain.callbacks import get_openai_callback
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings, OpenAIEmbeddings

dotenv.load_dotenv()

# Load environment variables (e.g., BASIC_AUTH_USERNAME, BASIC_AUTH_PASSWORD)

# Initialize different embeddings based on the user's selection
embedding_options = {
    "Hugging Face": HuggingFaceEmbeddings(),
    "OpenAI": OpenAIEmbeddings(),
    # "Hugging Face Instruct": HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl", model_kwargs={"device": "cpu"})
}

# Get the user's selected embedding option using radio buttons
selected_embedding = st.sidebar.radio("Select Embedding", list(embedding_options.keys()))

# Initialize the chosen embedding
embeddings = embedding_options[selected_embedding]
print("Embeddings type: ", type(embeddings).__name__.lower())

faiss_index = None

# Helper function to load FAISS index
def load_faiss_index(embeddings):
    # Get the directory name based on the selected embedding
    embedding_type = type(embeddings).__name__.lower()
    faiss_index_path = f"faiss_index_{embedding_type}"
    # index file path
    print("Searching faiss: ", faiss_index_path)

    global faiss_index
    if faiss_index is not None:
        print(f"FAISS index already loaded {faiss_index}.")
        return faiss_index
    else:
        if os.path.exists(faiss_index_path):
            # Load the FAISS index from the specified directory
            faiss_index = FAISS.load_local(faiss_index_path, embeddings)
            return faiss_index
        else:
            print(f"FAISS index directory for {embedding_type} not found.")
            return None
    
# Load the existing FAISS index
faiss_index = load_faiss_index(embeddings)

# Prompt template for Hugging Face Instruct Embeddings
temp_prompt = """Given the following extracted parts of a long document known as context and a question, 
create a final answer with references ("SOURCES") from the part TITLE. The TITLE is started after the keyword TITLE and ends before SLUG. 
If you don't know the answer, just say that you don't know. Don't try to make up an answer and make sure the answers are constructed from the context.
ALWAYS return a "SOURCES" part in your answer. LEAD_PARA and PARA have high weightage. So, use the context thoroughly and give the final answer as a summarized content.

{context}

Question: {question}
Final Answer in English:
SOURCE:
"""

# Function to create and store vectors in the FAISS index
def create_and_store_vectors(articles, embeddings):
    # Get the directory name based on the selected embedding
    embedding_type = type(embeddings).__name__.lower()
    faiss_index_path = f"faiss_index_{embedding_type}"

    if not os.path.exists(faiss_index_path):
        # If the FAISS index directory for the selected embedding doesn't exist, create it
        os.makedirs(faiss_index_path)

    source_chunks = []
    for article in articles:
        # Assuming article['content'] contains the text of the article
        content = article['content']

        # Preprocess the article content (you can customize this)
        preprocessed_content = preprocess(content)

        # Generate embeddings for the article
        article_embedding = embeddings.embed(preprocessed_content)

        # Create a Document object with the article's embedding and metadata (article_id)
        document = Document(vector=article_embedding, metadata={"id": article['article_id']})

        # Append the Document to the list
        source_chunks.append(document)

    # Create a new FAISS index
    faiss_index = FAISS.from_documents(source_chunks, embeddings)

    # Save the FAISS index to disk
    faiss_index.save_local(os.path.join(faiss_index_path, "faiss_index"))

    # Save metadata (embedding type) to a config file
    with open(os.path.join(faiss_index_path, "config.json"), "w") as config_file:
        config = {"embedding_type": embedding_type}
        json.dump(config, config_file)

# Function to update and reload vectors in the FAISS index
def update_and_reload_vectors(article_content, article_id, embeddings, faiss_index_path):
    # Assuming article_content is the updated text of the article
    preprocessed_content = preprocess(article_content)

    # Generate embeddings for the updated article content
    updated_embedding = embeddings.embed(preprocessed_content)

    # Create a Document object with the updated article's embedding and metadata (article_id)
    updated_document = Document(vector=updated_embedding, metadata={"id": article_id})

    # Load the existing FAISS index
    faiss_index = load_faiss_index(faiss_index_path, embeddings)

    if faiss_index is not None:
        # Get the index of the existing document with the same article_id
        doc_index = faiss_index.asimilarity_search_by_vector(updated_document, k=1)[0]

        # Update the existing document with the updated embedding
        faiss_index.replace(doc_index, updated_document)

        # Save the updated index to disk
        faiss_index.save_local(faiss_index_path)

# Function to get answers using the selected embedding and FAISS index
def get_answeres(question, embeddings, faiss_index):
    # Fill the prompt template based on the selected embedding
    prompt_template = PromptTemplate(template=temp_prompt, input_variables=["context", "question"])

    qa_chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0), chain_type="stuff", retriever=faiss_index.as_retriever(), chain_type_kwargs={"prompt": prompt_template}) # this method reloads the index so application fails to run

    with get_openai_callback() as cb:
        answeres = qa_chain.run(question)
        print("Answers: ", answeres)
        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Prompt Tokens: {cb.prompt_tokens}")
        print(f"Completion Tokens: {cb.completion_tokens}")
        print(f"Total Cost (USD): ${cb.total_cost}")

    return answeres

# Preprocess function (you can customize this as per your data preprocessing needs)
def preprocess(text):
    # Your implementation to clean and preprocess the text data
    return text

# Streamlit app setup
st.title("Trippy Bot: Answers your questions with the knowledge of https://trip101.com")
st.sidebar.header("Settings")

# Streamlit app main section
question = st.text_input("Enter your question:")

# Button to get answers
if st.button("Get Answers"):
    if question:
        # Call the function to get answers and highlight usage cost messages
        answeres = get_answeres(question, embeddings, faiss_index)
        if answeres:
            st.write("Answer:", answeres)
        else:
            st.write("Error:", "Something went wrong! Please try after some time.")

    else:
        st.warning("Please enter a question.")
