import os

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

from google.adk.agents import Agent, LlmAgent
from .custom_agent import SelfRagAgent
from .prompts import (
    GENERATE_INSTRUCTION,
    GRADE_DOCUMENT_INSTRUCTION,
    QUERY_REWRITER_INSTRUCTION,
    RETRIEVER_INSTRUCTION,
    HALLUCINATION_CHECK_INSTRUCTION,
    RELEVANCE_CHECK_INSTRUCTION,
)
from .tools.tools import PineconeIndexRetrieval

load_dotenv()

openai = OpenAI()


def get_embedding(text: str) -> list[float]:
    embedding = openai.embeddings.create(model="text-embedding-3-large", input=text)
    return embedding.data[0].embedding


pinecone_tool = PineconeIndexRetrieval(
    name="pinecone_retrieval_tool",
    description="This tool retrieves data from the pinecone vector database.",
    index_name=os.environ.get("PINECONE_INDEX_NAME"),
    namespace=os.environ.get("PINECONE_NAMESPACE"),
    pinecone=Pinecone(),
    embedder=get_embedding,
    top_k=5,
)

retriever = LlmAgent(
    name="Retriever",
    model="gemini-2.0-flash",
    description="This tool retrieves data from the pinecone vector database.",
    instruction=RETRIEVER_INSTRUCTION,
    tools=[pinecone_tool],
    output_key="retriever_result"
)

grade_document = LlmAgent(
    name="GradeDocument",
    model="gemini-2.0-flash",
    description="This tool grades documents based on their relevance to a given query and user input.",
    instruction=GRADE_DOCUMENT_INSTRUCTION,
    output_key="grade_document_result"
)

query_rewriter = LlmAgent(
    name="QueryRewriter",
    model="gemini-2.0-flash",
    description="This tool rewrites queries to improve their relevance.",
    instruction=QUERY_REWRITER_INSTRUCTION,
    output_key="query"
)

generate = LlmAgent(
    name="Generate",
    model="gemini-2.0-flash",
    description="This tool generates answers based on the retrieved documents and user input.",
    instruction=GENERATE_INSTRUCTION,
    output_key="generate_result"
)

hallucination_checker = LlmAgent(
    name="HallucinationChecker",
    model="gemini-2.0-flash",
    description="This tool checks for hallucinations in generated answers.",
    instruction=HALLUCINATION_CHECK_INSTRUCTION,
    output_key="hallucination_check_result"
)

relevance_check = LlmAgent(
    name="RelevanceCheck",
    model="gemini-2.0-flash",
    description="This tool checks the relevance of generated answers.",
    instruction=RELEVANCE_CHECK_INSTRUCTION,
    output_key="relevance_check_result"
)

root_agent = SelfRagAgent(
    name="SelfRAGAgent",
    description="This agent performs self-retrieval-augmented generation.",
    retriever=retriever,
    document_grader=grade_document,
    query_rewriter=query_rewriter,
    generator=generate,
    hallucination_checker=hallucination_checker,
    relevence_checker=relevance_check,
    output_key="self_rag_result",
)
