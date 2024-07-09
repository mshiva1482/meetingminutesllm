from dotenv import load_dotenv
import os
from langchain_community.llms import Ollama
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever

import shutil
import ollama

load_dotenv()

from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader

import boto3
from botocore.exceptions import ClientError

CHROMA_PATH = 'chroma_data'

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

def main():

    # Load documents
    loader = DirectoryLoader('data/mom/', glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    extract_action_items(documents=documents)

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

def send_email(action_items, recipient_email):
    ses_client = boto3.client('sns')
    sender_email = os.environ['SENDER_EMAIL']

    # Create email HTML
    html = "<html><body>"
    html += "<h1>Action Items from Meeting</h1>"
    html += "<ul>"
    
    for item in action_items:
        html += f"<li>{item['action']} on {item['date']}</li>"
    
    html += "</ul>"
    html += "</body></html>"
    
    # Prepare the email message
    subject = "Meeting Action Items"
    body_text = "Please view this email in an HTML-compatible email viewer."

    try:
        response = ses_client.send_email(
            Source=sender_email,
            Destination={
                'ToAddresses': [recipient_email]
            },
            Message={
                'Subject': {
                    'Data': subject
                },
                'Body': {
                    'Text': {
                        'Data': body_text
                    },
                    'Html': {
                        'Data': html
                    }
                }
            }
        )
        
        print(response)
    except ClientError as e:
        print(f"An error occurred: {e.response['Error']['Message']}")