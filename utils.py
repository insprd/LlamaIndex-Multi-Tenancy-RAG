import streamlit as st
from llama_index import (ServiceContext, SimpleDirectoryReader,
                         VectorStoreIndex, StorageContext,
                         load_index_from_storage)
from llama_index.llms import OpenAI
from llama_index.ingestion import IngestionPipeline
from llama_index.text_splitter import SentenceSplitter


# Function to add a new user
def add_new_user(new_user_input):
  new_user = new_user_input.strip()
  if new_user and new_user not in st.session_state.options:
    # Use the new option name as the code (or modify as needed)
    st.session_state.options[new_user] = new_user
    # Update the selectbox options
    st.session_state.selectbox_options.append(new_user)


# Function to load the data
def load_data(filepath):
  """
    Loads data from a file path.

    Returns:
    - A list of document objects.
  """
  with st.spinner(text="Loading the document"):
    reader = SimpleDirectoryReader(input_files=[filepath])
    docs = reader.load_data()
  return docs


# Function to create the index
def create_index():
  """
    Creates an empty index.
    Returns:
    - A VectorStoreIndex object.
  """
  service_context = ServiceContext.from_defaults(
      llm=OpenAI(temperature=0.1, model="gpt-3.5-turbo"))
  index = VectorStoreIndex.from_documents(documents=[],
                                          service_context=service_context)
  # Save the index to disk
  index.storage_context.persist(persist_dir="storage")


# Function to insert documents into index
def insert_documents(documents, user):
  """
    Inserts the documents into the index.
  """
  with st.spinner(text="Inserting documents into the index"):
    # Load the index from the storage context
    storage_context = StorageContext.from_defaults(persist_dir="storage")
    index = load_index_from_storage(storage_context)
    # Insert the documents into the index
    for document in documents:
      document.metadata['user'] = user
    pipeline = IngestionPipeline(transformations=[
        SentenceSplitter(chunk_size=512, chunk_overlap=20),
    ])
    nodes = pipeline.run(documents=documents)
    index.insert_nodes(nodes)
    # Save the index to disk
    index.storage_context.persist(persist_dir="storage")
