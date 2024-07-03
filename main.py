from dotenv import load_dotenv
import os
load_dotenv()

from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader

api_key = os.getenv("LLAMA_PARSE_API_KEY")

parser = LlamaParse(
    api_key=api_key,
    result_type="markdown"
)

file_extractor = {".pdf": parser}
documents = SimpleDirectoryReader(input_files=['data/mom/tnega.pdf'], file_extractor=file_extractor).load_data()
print(documents)

