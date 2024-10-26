from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from typing import Optional

# Import necessary langchain modules
from langchain_community.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferWindowMemory
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize LLM, memory, and index
llm = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))
memory = ConversationBufferWindowMemory(k=5)

# Create embeddings and text splitter
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)

# Load document and create index
try:
    loader = TextLoader("hospital.txt")
except Exception as e:
    print("Error while loading file=", e)

index_creator = VectorstoreIndexCreator(embedding=embedding, text_splitter=text_splitter)
index = index_creator.from_loaders([loader])

# Initialize FastAPI
# app = FastAPI()   



while True:
    human_message = input("How i can help you today? ")
    response = index.query(human_message, llm=llm, memory=memory)
    print(response)

# # Pydantic model for input validation
# class QueryRequest(BaseModel):
#     question: str
#     context: Optional[str] = None

# # FastAPI route to handle user queries
# @app.post("/query/")
# async def query_index(request: QueryRequest):
#     try:
#         response = index.query(request.question, llm=llm, memory=memory)
#         return {"response": response}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error querying the index: {e}")

# # Add Uvicorn server to run on localhost
# if __name__ == "__main__":
    
#     uvicorn.run(app, host="127.0.0.1", port=8000)
