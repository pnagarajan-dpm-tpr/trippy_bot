import os
import re
import json
import dotenv
import tiktoken
import logging
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

# Initialize the logger
logging.basicConfig(filename="app_log.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load environment variables (e.g., BASIC_AUTH_USERNAME, BASIC_AUTH_PASSWORD)

# Initialize different embeddings based on the user's selection
embedding_options = {
    "Hugging Face": HuggingFaceEmbeddings(),
    "OpenAI": OpenAIEmbeddings(),
    "Hugging Face Instruct": HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl", model_kwargs={"device": "cpu"})
}

# Helper function to load FAISS index
def load_faiss_index(embeddings):
    # Get the directory name based on the selected embedding
    embedding_type = type(embeddings).__name__.lower()
    faiss_index_path = f"faiss_index_{embedding_type}"
    # index file path
    logging.info("Searching FAISS index: ", faiss_index_path)

    global faiss_index
    if faiss_index is not None:
        logging.info(f"FAISS index already loaded {faiss_index}.")
        return faiss_index
    else:
        if os.path.exists(faiss_index_path):
            # Load the FAISS index from the specified directory
            faiss_index = FAISS.load_local(faiss_index_path, embeddings)
            return faiss_index
        else:
            logging.info(f"FAISS index directory for {embedding_type} not found.")
            return None
    
# Load the existing FAISS index
selected_embedding = st.sidebar.radio("Select Embedding", list(embedding_options.keys()))
embeddings = embedding_options[selected_embedding]
logging.info("Embeddings type: ", type(embeddings).__name__.lower())

faiss_index = load_faiss_index(embeddings)

# Prompt template for Hugging Face Instruct Embeddings
temp_prompt = """Given the following extracted parts of a long document known as context and a question, 
create a final answer with references to ("SOURCES"), SOURCES should contain following parts "Title: " the title part is enclosed inside [bot-data-title]...[/bot-data-title], 
and "Link: " use the part Slug which is enclosed inside [bot-data-slug]..[/bot-data-slug] and replace the Slug part in the following URL https://trip101.com/article/Slug.
If you don't know the answer, just say that you don't know. Don't try to make up an answer and make sure the answers are constructed from the context.
ALWAYS return a "SOURCES" part in your answer. As per the context weightage lead_para which is enclosed within [bot-data-lead-para]...[/bot-data-lead-para] has high weightage, 
and we have multiple paras which enclosed with in [bot-data-para]...[/bot-data-para], Each para has enclosed within the tags. 
Gather information as per the weightage don't miss any para and give the final answer short and crisp, make sure the domain you concat in the Link is always "https://trip101/article/"
make sure the answer is formatted to the question asked and should not be out of the context provided to you.

{context}

Question: {question}
Final Answer in English:
SOURCES:
"""


# Function to get answers using the selected embedding and FAISS index
def get_answers(question, embeddings, faiss_index):
    prompt_template = PromptTemplate(template=temp_prompt, input_variables=["context", "question"])

    qa_chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0), chain_type="stuff", retriever=faiss_index.as_retriever(), chain_type_kwargs={"prompt": prompt_template})

    with get_openai_callback() as cb:
        answeres = qa_chain.run(question)
        logging.info(f"Answers          : {answeres}")
        logging.info(f"Total Tokens     : {cb.total_tokens}")
        logging.info(f"Prompt Tokens    : {cb.prompt_tokens}")
        logging.info(f"Completion Tokens: {cb.completion_tokens}")
        logging.info(f"Total Cost (USD) : ${cb.total_cost}")

    return answeres

# Streamlit app setup
logging.info("Trippy bot application is being loaded...!")
st.title("Trippy Bot: Answers your questions with the knowledge of https://trip101.com")
logging.info("Trippy bot application's settings is being loaded...!")
st.sidebar.header("Settings")

# Function to get the session state
def get_session_state():
    if not hasattr(st, '_custom_session_state'):
        st._custom_session_state = {}
    return st._custom_session_state

# Display the history of questions and answers
st.subheader("Question History")
session_state = get_session_state()
if 'history' in session_state:
    for entry in session_state['history']:
        st.write("Question:", entry["question"])
        st.write("Answer:", entry["answer"])
        st.write("-" * 50)

# Streamlit app main section
question = st.text_input("Enter your question:")

# Button to get answers
if st.button("Get Answers"):
    if question:
        # Call the function to get answers and highlight usage cost messages
        answers = get_answers(question, embeddings, faiss_index)
        if answers:
            st.write("Answer:", answers)

            # Store the question and answer in the history
            session_state = get_session_state()
            if 'history' not in session_state:
                session_state['history'] = []
            session_state['history'].append({"question": question, "answer": answers})
        else:
            st.write("Error:", "Something went wrong! Please try again later.")
    else:
        st.warning("Please enter a question.")