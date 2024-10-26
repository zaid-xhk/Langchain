from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools import Tool
from langchain import hub
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.text_splitter import CharacterTextSplitter
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from fastapi.middleware.cors import CORSMiddleware


import os
import requests
from dotenv import load_dotenv
import uvicorn  # Import uvicorn

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the LLM with GoogleGenerativeAI
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Initialize Tavily search tool
search = TavilySearchResults(tavily_api_key=os.getenv("TAVILY_API_KEY"))

# Define request model
class ChatRequest(BaseModel):
    session_id: str
    user_input: str

# Setup the RAG model and FAISS VectorStore
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=200)

try:
    loader = TextLoader("hospital.txt")
    index_creator = VectorstoreIndexCreator(
        embedding=embedding, 
        vectorstore_cls=FAISS,
        text_splitter=text_splitter
    )
    index = index_creator.from_loaders([loader])
except Exception as e:
    print("Error while loading or indexing the document:", e)
    index = None  # Set to None if there's an issue

# Function to query the RAG model
def rag_query(user_input: str) -> str:
    if not index:
        return "Document index is not available."
    response = index.query(user_input, llm=llm)
    return response or "No relevant information found in the document."

# RAG tool definition
rag_tool = Tool(
    name="rag_query",
    func=rag_query,
    description="Queries the RAG model for Delta Dev hospital related info"
)

# Function to book an appointment
def book_appointment(user_id: str, doctor: str, date: str, time: str) -> str:
    url = "http://localhost:8000/book_appointment/"  # FastAPI endpoint
    data = {
        "user_id": user_id,
        "doctor": doctor,
        "date": date,
        "time": time
    }
    
    try:
        # Placeholder response for demonstration
        return "Appointment booked successfully!"  # Placeholder message
    except requests.exceptions.RequestException as e:
        return f"Error booking appointment: {str(e)}"

# Appointment booking tool
book_appointment_tool = Tool(
    name="book_appointment",
    func=book_appointment,
    description="Books an appointment with a specified doctor, date, and time."
)

# List of tools to be used by the agent
tools = [search, book_appointment_tool, rag_tool]

# Pull the prompt from LangChain hub
prompt = hub.pull("hwchase17/openai-functions-agent")

# Create the tool-calling agent
agent = create_tool_calling_agent(llm, tools, prompt)

# Create the agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# In-memory chat message history
message_history = ChatMessageHistory()

# Runnable with chat history
agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    lambda session_id: message_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

# FastAPI endpoint for chatbot interaction
@app.post("/chat")
async def chat(request: ChatRequest):
    print("request recieved")
    try:
        # Invoke the agent with chat history and return the response
        response = agent_with_chat_history.invoke(
            {"input": request.user_input},
            config={"configurable": {"session_id": request.session_id}}
        )
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the app using uvicorn when the script is executed directly
if __name__ == "__main__":
    uvicorn.run("integeration:app", host="127.0.0.1", port=8000, reload=True)