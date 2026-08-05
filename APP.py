from fastapi import FastAPI
from langserve import add_routes

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key="YOUR_GOOGLE_API_KEY"
)

# Prompt Template
prompt = ChatPromptTemplate.from_template(
    "Answer the following question:\n\n{question}"
)

# Output Parser
parser = StrOutputParser()

# Create LangChain pipeline
chain = prompt | llm | parser

# Create FastAPI app
app = FastAPI(
    title="LangServe Demo",
    version="1.0",
    description="Simple LangServe + FastAPI Example"
)

# Add LangServe route
add_routes(
    app,
    chain,
    path="/chat"
)

# Root endpoint
@app.get("/")
def home():
    return {"message": "LangServe API is Running"}
