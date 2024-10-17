import os
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

import argparse
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_community.embeddings import OpenAIEmbeddings

import requests
from fpdf import FPDF
from bs4 import BeautifulSoup
from openai import OpenAI
import getpass

client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

# Set up the OpenAI API client


def crawl_webpage(url):
    response = requests.get(url)

    print(f"Response Status Code: {response.status_code} for {url}")
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text_content = ' '.join([para.get_text() for para in paragraphs])
        return text_content
    else:
        return None


# List of websites to crawl
urls = [
    'https://www.techtarget.com/whatis/definition/pseudoscience#:~:text=The%20term%20pseudoscience%20can%20refer,stand%20up%20under%20scientific%20scrutiny.'
]


def save_to_pdf(text, filename):
    # Create a PDF instance
    pdf = FPDF()

    pdf.set_auto_page_break(auto=True, margin=15)

    # Set font
    pdf.set_font("Arial", size=12)

    # Example text stored in a variable
    my_text = website_text

    # Encode the text to UTF-8 and then decode it to Latin-1, replacing unencodable characters
    # with a replacement character (e.g., �)
    encoded_text = my_text.encode('utf-8', 'replace').decode('latin-1')

    # Add a page
    pdf.add_page()

    # Add the encoded text to the PDF using the variable
    pdf.multi_cell(0, 10, encoded_text)

    # Save the PDF
    pdf.output(filename)


for i, url in enumerate(urls):
    website_text = crawl_webpage(url)
    for i, url in enumerate(urls):
        website_text = crawl_webpage(url)
        if website_text:
            # Save the extracted text to a PDF
            filename = os.path.join("data/", f'website_content_{i+1}.pdf')
            save_to_pdf(website_text, filename)
            print(f"Content from {url} saved to {filename}")
        else:
            print(f"Failed to retrieve content from {url}")


def load_document2():
    document_loader = PyPDFLoader(DATA_PATH)
    return document_loader.load()


###################################################################
# Ensure the OpenAI API key is set
my_secret = os.environ['OPENAI_API_KEY']

#set the app title
st.title('PseudoScience app')

st.write('Welcome to PseudoScience app. Ask me anything!')  # Initial text

# User inputs the question
question = st.text_input("Enter your questions:")

if st.button("Enter"):
    CHROMA_PATH = "chroma"
    PROMPT_TEMPLATE = """
    Answer the question based only on the following context:

    {context}

    ---

    Answer the question based on the above context: {question}
    """

    def main(query_text=question):
        # Prepare the DB.
        embedding_function = OpenAIEmbeddings(
            api_key=os.environ['OPENAI_API_KEY'])
        db = Chroma(persist_directory=CHROMA_PATH,
                    embedding_function=embedding_function)

        # Search the DB.
        results = db.similarity_search_with_relevance_scores(query_text, k=3)
        if len(results) == 0 or results[0][1] < 0.7:
            st.write("Unable to find matching results.")
            return

        context_text = "\n\n---\n\n".join(
            [doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text,
                                        question=query_text)

        # Generate a response
        model = ChatOpenAI(api_key=os.environ['OPENAI_API_KEY'])
        response_text = model.predict(prompt)

        sources = [doc.metadata.get("source", None) for doc, _score in results]
        formatted_response = f"Response: {response_text}\nSources: {sources}"
        st.write(formatted_response)

    # Run the main function
    main()


#get document
def get_embedding_function():
    return OpenAIEmbeddings(api_key=os.environ['OPENAI_API_KEY'])


CHROMA_PATH = "chroma"

if not os.path.exists(CHROMA_PATH):
    os.makedirs(CHROMA_PATH)
DATA_PATH = "data"


def main2():

    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset",
                        action="store_true",
                        help="Reset the database.")
    args = parser.parse_args(args=[])
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Create (or update) the data store.
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)


def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    db = Chroma(persist_directory=CHROMA_PATH,
                embedding_function=get_embedding_function())

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("✅ No new documents to add")


def calculate_chunk_ids(chunks):

    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main2()
