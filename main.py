import os
import streamlit as st
import pickle
import time
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS



from dotenv import load_dotenv
load_dotenv() # take environment variables from .env
from secret_key import grok_key
os.environ['GROQ_API_KEY'] = grok_key

st.title("News Research Tool ")

st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

file_path = "FAISS_store_grok.pkl"

main_placefolder = st.empty()

llm = ChatGroq(temperature=0.6, model_name="llama-3.3-70b-versatile",api_key=grok_key)

if process_url_clicked:
    #load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placefolder.text("Data Loading....Started....:)")
    data = loader.load()
    # Split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n",'\n','.',','],
        chunk_size=1000
    )
    main_placefolder.text("Text Splitter....Started....:)")
    docs = text_splitter.split_documents(data)
    #create embeddings and save it to FAISS index
    embeddings = HuggingFaceEmbeddings()
    vectorstore_grok = FAISS.from_documents(docs,embeddings)
    main_placefolder.text("Embedding Vector Started Building....:)")
    time.sleep(2)

    # Save the FAISS index to a pickle file
    with open (file_path,'wb') as f:
        pickle.dump(vectorstore_grok, f)

query = main_placefolder.text_input("Question: ")
if query:
        if os.path.exists(file_path):
            with open(file_path,'rb') as f:
                vectorstore = pickle.load(f)
                chain = RetrievalQAWithSourcesChain.from_llm(llm=llm,retriever= vectorstore.as_retriever())
                result = chain({'question':query},return_only_outputs=True)

                st.header('Answer')
                st.write(result['answer'])

                # Display sources if available
                sources = result.get('sources',"")
                if sources:
                    st.subheader('Sources:')
                    sources_list = sources.split('\n') #split the sources by newline
                    for source in sources_list:
                        st.write(source)