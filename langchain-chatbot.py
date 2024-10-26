from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools import Tool
from langchain import hub
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.text_splitter import CharacterTextSplitter
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.document_loaders import TextLoader
import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the LLM with GoogleGenerativeAI
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Tavily search tool
search = TavilySearchResults(tavily_api_key=os.getenv("TAVILY_API_KEY"))

# Function to interact with the RAG model (queries document index)
def rag_query(user_input: str) -> str:
    # Use the RAG model to query the indexed document
    response = index.query(user_input, llm=llm)
    return response or "No relevant information found in the document."

# Create the RAG tool as a LangChain Tool
rag_tool = Tool(
    name="rag_query",
    func=rag_query,
    description="Queries the RAG model for deltadev hospital related info "
)

# Setup the RAG model
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=200)

# Load document and create the vectorstore index
try:
    loader = TextLoader("hospital.txt")
except Exception as e:
    print("Error while loading file:", e)

index_creator = VectorstoreIndexCreator(embedding=embedding, text_splitter=text_splitter)
index = index_creator.from_loaders([loader])


# Function to interact with the FastAPI appointment booking API
def book_appointment(user_id: str, doctor: str, date: str, time: str) -> str:
    url = "http://localhost:8000/book_appointment/"  # FastAPI endpoint
    data = {
        "user_id": user_id,
        "doctor": doctor,
        "date": date,
        "time": time
    }
    
    try:
        # # response = requests.post(u, json=data)
        # if response.status_code == 200:
        #     return response.json().get("message", "Appointment booked successfully!")
        # else:
            return f"okkkkkkkkkkk "
    except requests.exceptions.RequestException as e:
        return f"Error: {str(e)}"

# Create a custom tool for LangChain
book_appointment_tool = Tool(
    name="book_appointment",
    func=book_appointment,
    description="Books an appointment with a specified doctor, date, and time."
)

# List of tools to be used by the agent
tools = [search, book_appointment_tool, rag_tool]

# Pull the prompt from LangChain hub
prompt = hub.pull("hwchase17/openai-functions-agent")

# Create the tool calling agent
agent = create_tool_calling_agent(llm, tools, prompt)

# Create the agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True )

# In-memory chat message history
message_history = ChatMessageHistory()

# Runnable with chat history
agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    lambda session_id: message_history,  # Return message history based on session_id
    input_messages_key="input",  # Key for input messages
    history_messages_key="chat_history"  # Key for history messages
)

# Start the chatbot loop
while True:
    user_input = input("How can I help you today? : ")
    agent_with_chat_history.invoke(
        {"input": user_input},  # Pass the user input
        config={"configurable": {"session_id": "test123"}}  # Use a test session_id
    )
