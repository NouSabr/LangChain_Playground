import os
from dotenv import load_dotenv
from langchain_unstructured import UnstructuredLoader

from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings

from langchain_chroma import Chroma

from uuid import uuid4

load_dotenv()

file_paths = [
    "/home/nouranabry/workspace/Learning/LangChain/LangChain_Playground/rag-gist/data/mediumblog1.txt",
]

loader = UnstructuredLoader(
    file_path=file_paths,
    api_key=os.getenv("UNSTRUCTURED_API_KEY"),
    url=os.getenv("UNSTRUCTURED_API_URL"),
    partition_via_api=True,
    chunking_strategy="basic",
    max_characters=1000000
    )

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

embeddings = AzureOpenAIEmbeddings(
    model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"), # type: ignore
    api_version=os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY"), # type: ignore
    dimensions=256
)

vector_store = Chroma(
    collection_name="blogs",
    embedding_function=embeddings,
    chroma_cloud_api_key=os.getenv("CHROMA_API_KEY"),
    tenant=os.getenv("CHROMA_TENANT"),
    database=os.getenv("CHROMA_DATABASE"),
)

if __name__ == '__main__':
    print("Ingesting Document...")

    document = loader.load()

    print("Splitting...")
    texts = text_splitter.split_text(document[0].page_content)
    documents = text_splitter.create_documents(texts)
    print(f"created {len(texts)} chunks")

    print("Ingesting Chunks...")

    uuids = [str(uuid4()) for _ in range(len(documents))]

    vector_store.add_documents(documents=documents, ids=uuids)

    print("Finished!")

