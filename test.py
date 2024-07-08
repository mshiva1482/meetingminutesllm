from langchain.retrievers import ParentDocumentRetriever

from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma

client = chromadb.HttpClient(host="localhost", port=8000, settings=Settings(allow_reset=True))

loaders = [
    TextLoader("data/textdocs/text1.txt"),
    TextLoader("data/textdocs/text2.txt"),
]

docs = []
for loader in loaders:
    docs.extend(loader.load())


child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

vectorstore = Chroma(
    client=client,
    collection_name="full_documents",
    embedding_function=OllamaEmbeddings(model="mxbai-embed-large")
)

store = InMemoryStore()
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
)

retriever.add_documents(docs, ids=None)

sub_docs = retriever.invoke("Emma Blackwood")
print(sub_docs[0].page_content)