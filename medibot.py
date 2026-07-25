import os
from urllib import response
import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile




## Uncomment the following files if you're not using pipenv as your virtual environment manager
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())




DB_FAISS_PATH="vectorstore/db_faiss"
@st.cache_resource
def get_vectorstore():
    embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def process_uploaded_pdf(uploaded_file):
    # Save uploaded file to a temp location so PyPDFLoader can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.from_documents(chunks, embedding_model)

    return db

def process_uploaded_pdf(uploaded_file):
    # Save uploaded file to a temp location so PyPDFLoader can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.from_documents(chunks, embedding_model)

    return db

def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


def load_llm(huggingface_repo_id, HF_TOKEN):
    llm=HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token":HF_TOKEN,
                      "max_length":"512"}
    )
    return llm


def main():
    st.title("Ask Chatbot!")
    uploaded_file = st.file_uploader("Upload a PDF to chat with (optional)", type="pdf")

    if uploaded_file is not None:
        if 'uploaded_db' not in st.session_state or st.session_state.get('uploaded_filename') != uploaded_file.name:
            with st.spinner("Processing your PDF..."):
                st.session_state.uploaded_db = process_uploaded_pdf(uploaded_file)
                st.session_state.uploaded_filename = uploaded_file.name
            st.success(f"'{uploaded_file.name}' is ready to query!")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt=st.chat_input("Pass your prompt here")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role':'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
                Use the pieces of information provided in the context to answer user's question.
                If you dont know the answer, just say that you dont know, dont try to make up an answer. 
                Dont provide anything out of the given context

                Context: {context}
                Question: {question}

                Start the answer directly. No small talk please.
                """
        
        #HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3" # PAID
        #HF_TOKEN=os.environ.get("HF_TOKEN")  

        #TODO: Create a Groq API key and add it to .env file
        
        
        try: 
            if 'uploaded_db' in st.session_state:
                vectorstore = st.session_state.uploaded_db
            else:
                vectorstore = get_vectorstore()

            if vectorstore is None:
                st.error("Failed to load the vector store")
                st.stop()

            qa_chain = RetrievalQA.from_chain_type(
                llm=ChatGroq(
                    model_name="openai/gpt-oss-120b",  # free, fast Groq-hosted model
                    temperature=0.0,
                    groq_api_key=os.environ["GROQ_API_KEY"],
                ),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k':3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response = qa_chain.invoke({'query': prompt})

            result = response["result"]
            source_documents = response["source_documents"]

            # Build a clean, readable source list instead of dumping raw objects
            sources_text = "\n\n**Sources:**\n"
            seen_pages = set()
            for doc in source_documents:
                source_file = doc.metadata.get("source", "Unknown source")
                page_label = doc.metadata.get("page_label", doc.metadata.get("page", "?"))
                key = (source_file, page_label)
                if key not in seen_pages:
                    seen_pages.add(key)
                    filename = source_file.split("\\")[-1].split("/")[-1]
                    sources_text += f"- {filename}, page {page_label}\n"

            result_to_show = result + sources_text
            st.chat_message('assistant').markdown(result_to_show)
            st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})

        except FileNotFoundError:
            st.error("⚠️ No document found. Please upload a PDF above to get started.")
        except KeyError:
            st.error("⚠️ Missing API key. Please make sure GROQ_API_KEY is set in your .env file.")
        except Exception as e:
            error_message = str(e)
            if "401" in error_message or "invalid_api_key" in error_message.lower():
                st.error("⚠️ Invalid API key. Please check your GROQ_API_KEY in the .env file.")
            elif "404" in error_message or "model_not_found" in error_message.lower():
                st.error("⚠️ The AI model is unavailable right now. Please try again later.")
            elif "rate_limit" in error_message.lower() or "429" in error_message:
                st.error("⚠️ Too many requests right now. Please wait a moment and try again.")
            else:
                st.error(f"⚠️ Something went wrong: {error_message}")
if __name__ == "__main__":
    main()