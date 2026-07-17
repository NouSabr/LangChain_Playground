import os
from dotenv import load_dotenv
from operator import itemgetter

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_chroma import Chroma

from langchain_core.documents import Document

load_dotenv()

print ("Initializing Components...")

llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  # type: ignore
)

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

retriever = vector_store.as_retriever(
    search_type = "similarity",
    k=3,
    )

prompt_template = ChatPromptTemplate.from_template(
    """
    Answer the following question based on only the following context:
    {context}

    Question: {question}

    Provide a detailed answer:

    """
)


def format_docs(docs: list[Document]) -> str:
    """Format retrieved documents into a single string"""
    return "\n\n".join(doc.page_content for doc in docs)

def retrieval_chain_without_lcel(query: str):
    """
    Sample retrieval chain without LCEL.
    Manually retrieves documents, formats them, and generates a response.
    
    """
    docs = retriever.invoke(query)

    string_docs = format_docs(docs)

    prompts = prompt_template.format_messages(context=string_docs, question=query)

    response = llm.invoke(input=prompts)

    return response.content

def create_retrieval_chain_with_lcel():
    """
    Create a retrieval chain with LCEL
    Returns a chain that can be invoked with {"question": "..."}

    """
    retrieval_chain = (
        RunnablePassthrough.assign(
                context = itemgetter("question")
            |   retriever
            |   format_docs
            )
        |   prompt_template
        |   llm
        |   StrOutputParser()
    )

    return retrieval_chain


if __name__ == '__main__':
    print("Retrieving...")
    
    query = "What is Pinecone in Machine Learning?"

    # Raw Invocation without RAG:

    print("\n" + "*" * 20)
    print ("IMPLEMENTATION 0: Raw LLM Invocation (No RAG)")
    print ("*" * 20)
    result_raw = llm.invoke([HumanMessage(content=query)])
    print("\nAnswer:")
    print(result_raw.content)


    print("\n" + "*" * 20)
    print ("IMPLEMENTATION 1: Without LCEL")
    print ("*" * 20)
    result_without_lcel = retrieval_chain_without_lcel(query = query)
    print("\nAnswer:")
    print(result_without_lcel)

    print("\n" + "*" * 20)
    print ("IMPLEMENTATION 2: With LCEL")
    print ("*" * 20)
    chain_with_lcel = create_retrieval_chain_with_lcel()

    result_with_lcel = chain_with_lcel.invoke({"question": query})
    print("\nAnswer:")
    print(result_with_lcel)
