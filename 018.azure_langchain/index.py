# coding: utf-8

# https://github.com/hwchase17/langchain/issues/2491

import json
import logging
import os
import re

import io, sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

import chromadb
from dotenv import load_dotenv
from fastapi.encoders import jsonable_encoder
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import RetrievalQA

load_dotenv()
logging.basicConfig(level=logging.DEBUG)

AZURE_DEPLOYMENT_NAME = os.environ["AZURE_DEPLOYMENT_NAME"]
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_KEY"] = os.environ["AZURE_OPEN_API_KEY"]
os.environ["OPENAI_API_BASE"] = os.environ["AZURE_API_BASE"]
os.environ["OPENAI_VERSION"] = "2023-03-15-preview"

llm = AzureChatOpenAI(
    client=None,
    deployment_name=AZURE_DEPLOYMENT_NAME,
    openai_api_base=os.environ["AZURE_API_BASE"],
    openai_api_version="2023-03-15-preview",
    openai_api_key=os.environ["AZURE_OPEN_API_KEY"],
    temperature=0,
    request_timeout=180,
)

ABS_PATH = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(ABS_PATH, "db")


def replace_newlines_and_spaces(text):
    # Replace all newline characters with spaces
    text = text.replace("\n", " ")

    # Replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text)

    return text


def get_documents():
    return PyPDFLoader("data/web3.pdf").load()


def init_chromadb():
    if not os.path.exists(DB_DIR):
        os.mkdir(DB_DIR)

    client_settings = chromadb.config.Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=DB_DIR,
        anonymized_telemetry=False,
    )

    embeddings = OpenAIEmbeddings(deployment="text-embedding-ada-002", chunk_size=1)

    vectorstore = Chroma(
        collection_name="langchain_store",
        embedding_function=embeddings,
        client_settings=client_settings,
        persist_directory=DB_DIR,
    )

    documents = []
    for num, doc in enumerate(get_documents()):
        doc.page_content = replace_newlines_and_spaces(doc.page_content)
        documents.append(doc)

    vectorstore.add_documents(documents=documents, embedding=embeddings)
    vectorstore.persist()
    print(vectorstore)

    documents = []
    for num, doc in enumerate(PyPDFLoader("data/shingikai.pdf").load()):
        doc.page_content = replace_newlines_and_spaces(doc.page_content)
        documents.append(doc)

    vectorstore.add_documents(documents=documents, embedding=embeddings)
    vectorstore.persist()
    print(vectorstore)


def query_chromadb():
    if not os.path.exists(DB_DIR):
        raise Exception(f"{DB_DIR} does not exist, nothing can be queried")

    client_settings = chromadb.config.Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=DB_DIR,
        anonymized_telemetry=False,
    )

    embeddings = OpenAIEmbeddings()

    vectorstore = Chroma(
        collection_name="langchain_store",
        embedding_function=embeddings,
        client_settings=client_settings,
        persist_directory=DB_DIR,
    )

    retriever = vectorstore.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    # query = "中長期のミッションステートメントは？"
    query = "令和3年6月7日の消費経済審議会で何の話をしましたか？"
    result = qa.run(query)

    # result = vectorstore.similarity_search_with_score(query="web3とはなんですか？", k=1)
    # jsonable_result = jsonable_encoder(result)
    # print(json.dumps(jsonable_result, indent=2))
    print(result, file=sys.stderr)


def main():
    # init_chromadb()
    query_chromadb()


if __name__ == "__main__":
    main()
