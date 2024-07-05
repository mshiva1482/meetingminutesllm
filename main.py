from dotenv import load_dotenv
import os
from langchain_community.llms import Ollama
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import ChatPromptTemplate
import shutil
import ollama

load_dotenv()

from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader

PROMPT_TEMPLATE = '''
        Given the following text extracted from meeting minutes, generate a list of action items, their associated dates (if any) and the person/entity associated with the action item in JSON format. Each action item should be an object with the following properties:
        - "action": The specific task or action to be taken
        - "date": The due date or relevant date for the action (if mentioned), formatted as YYYY-MM-DD. If no date is specified, use null.
        - "entity": The person/group associated with the action item. If no person is associated, use null.

        Present the results as a JSON array of these objects. Ensure that each action item is clear, concise, and actionable. Ignore general discussion points or decisions that don't require specific actions.

        Meeting minutes text:

        {context}

        Please provide the JSON output of action items based on this text. If there are no action items in the meeting minutes, then return "None".
    '''

api_key = os.getenv("LLAMA_PARSE_API_KEY")
CHROMA_PATH = 'chroma_data'

def main():
    documents = document_parser(api_key)
    generate_embeddings(documents)

def document_parser(api_key):
    parser = LlamaParse(
        api_key=api_key,
        result_type="markdown"
    )

    file_extractor = {".pdf": parser}
    documents = SimpleDirectoryReader(input_files=['data/mom/tnega.pdf'], file_extractor=file_extractor).load_data()
    return(documents)

def generate_embeddings(documents: list[Document]):
    document = [doc.get_content() for doc in documents]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=80,
        chunk_overlap=25,
        length_function=len,
        add_start_index=True,
    )

    chunks = text_splitter.create_documents(document)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    
    db = Chroma.from_documents(
        chunks, OllamaEmbeddings(model="mxbai-embed-large"), persist_directory=CHROMA_PATH
    )

    # db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

def generate_summary(documents, model_name="llama3", temperature=0.7):
    text = "\n\n".join(doc.get_content() for doc in documents)

    llm = Ollama(
        model=model_name,
        temperature=temperature
    )

    # Create a Document object from the input text
    doc = Document(page_content=text)

    # Load the summarization chain
    chain = load_summarize_chain(llm, chain_type="stuff")

    # Generate the summary
    summary = chain.invoke([doc])

    print("Meeting summary: \n")
    print(summary)

def extract_action_items(documents):
    text = "\n\n".join(doc.get_content() for doc in documents)
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=text)

    output = ollama.generate(
        model="llama3",
        prompt=prompt
    )

    print("Extracted action items: \n")
    print(output['response'])


    

if __name__ == "__main__":
    documents = document_parser(api_key)
    
    # Generate embeddings for interacting with meeting minutes
    generate_embeddings(documents)
    
    # Generate summary
    generate_summary(documents)

    #Extract Action items
    extract_action_items(documents)