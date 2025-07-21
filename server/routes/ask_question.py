from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse
from modules.llm import get_llm_chain
from modules.query_handlers import query_chain
from langchain_pinecone import Pinecone as PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from logger import logger
import os

router = APIRouter()

@router.post("/ask/")
async def ask_question(question: str = Form(...)):
    try:
        logger.info(f"user query: {question}")

        # Embed model
        embed_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        # Pinecone vector store
        vectorstore = PineconeVectorStore.from_existing_index(
            index_name=os.environ["PINECONE_INDEX_NAME"],
            embedding=embed_model
        )
        retriever = vectorstore.as_retriever()
        chain = get_llm_chain(retriever)
        result = query_chain(chain, question)

        logger.info("query successful")
        return result

    except Exception as e:
        logger.exception("Error processing question")
        return JSONResponse(status_code=500, content={"error": str(e)})