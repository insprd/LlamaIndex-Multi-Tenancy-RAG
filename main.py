import os
import sys
import streamlit as st
from utils import *
from llama_index.vector_stores.types import MetadataFilters, ExactMatchFilter
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index import get_response_synthesizer

if 'OPENAI_API_KEY' not in os.environ:
  sys.stderr.write("""
  You haven't set up your API key yet.

  If you don't have an API key yet, visit:

  https://platform.openai.com/signup

  1. Make an account or sign in
  2. Click "View API Keys" from the top right menu.
  3. Click "Create new secret key"

  Then, open the Secrets Tool and add OPENAI_API_KEY as a secret.
  """)
  exit(1)

st.set_page_config(page_title="LlamaIndex Multi-Tenancy RAG",
                   page_icon="🦙",
                   layout="centered",
                   initial_sidebar_state="auto",
                   menu_items=None)

st.title("LlamaIndex 🦙 Multi-Tenancy RAG")

# Initialize session state for options if not already set
if 'options' not in st.session_state:
  st.session_state.options = {"Jerry": "Jerry"}

if 'selectbox_options' not in st.session_state:
  st.session_state.selectbox_options = list(st.session_state.options.keys())

# Sidebar for option selection and adding new options
with st.sidebar:
  selected_option = st.selectbox("Choose a User:",
                                 st.session_state.selectbox_options)

  # Input for adding new user
  new_user_input = st.text_input("Add New User")
  st.button("Add New User", on_click=lambda: add_new_user(new_user_input))

# Example of using the selected option's code
selected_user = st.session_state.options.get(selected_option, "Jerry")

# Create an Empty Index for the first time
if not os.path.exists("storage"):
  os.mkdir("storage")
  create_index()

# This is the main function running the app
if __name__ == "__main__":

  # Upload document option
  uploaded_file = st.file_uploader("Upload a document",
                                   type=["pdf", "txt", "docx"])
  # Check if a file is uploaded
  if uploaded_file is not None:
    file_path = uploaded_file.name
    with open(file_path, "wb") as f:
      f.write(uploaded_file.getvalue())
    documents = load_data(file_path)
    # print(documents)
    st.write("documents loaded")
    insert_documents(documents, selected_user)
    st.write("documents indexed")
  # Take input from the user
  user_input = st.text_input("Enter Your Query", "")
  storage_context = StorageContext.from_defaults(persist_dir="storage")
  # Display the input
  if st.button("Submit"):
    st.write(f"Your Query: {user_input}")
    with st.spinner("Thinking..."):
      # Load the index from the storage context
      index = load_index_from_storage(storage_context)
      retriever = VectorIndexRetriever(
          index=index,
          filters=MetadataFilters(
              filters=[ExactMatchFilter(
                  key="user",
                  value=selected_user,
              )]),
          similarity_top_k=3)
      response_synthesizer = get_response_synthesizer(response_mode="compact")
      query_engine = RetrieverQueryEngine(
          retriever=retriever, response_synthesizer=response_synthesizer)
      # Query the index
      result = query_engine.query(user_input)
      # Display the results
      st.write(f"Answer: {str(result)}")
