from fastapi import FastAPI, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi import UploadFile, File
from typing import Optional, Dict, List
import os
import json
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import PyPDF2
import uuid
from pydantic import BaseModel
from typing import List

app = FastAPI()
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

vector_stores: Dict[str, List[str]] = {}

# Global variable to store the LLMChain instance
llm_chain = None

class Document:
    def __init__(self, text):
        self.page_content = text
        self.metadata = {}

def get_vectorstore_from_text(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    document = Document(text)
    document_chunks = text_splitter.split_documents([document])
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_documents(document_chunks, embeddings)
    vector_store.save_local("faiss_index")
    return {"document_chunks": document_chunks, "embeddings": embeddings}
   
def get_mcq_chain() -> ChatGoogleGenerativeAI:
    mcq_model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.4)
    template = """{{Generate MCQs in JSON format:
    Create MCQ based on the context:\n {context}?\n
    The MCQ should be sent back in json format. 
    The JSON format should be like:
    "mcqs": [
            {{
                "question": "?",
                "options": [
                    "",
                    "",
                    "",
                    ""
                ],
                "correctAnswer": ""
            }},{{}},{{}} ... ]
    This json should strictly follow the format above. Do not deviate from the format.
    Only send the json back. Do not write anything else. Nothing else other than json. Peroid. 
    Generate 5-10 MCQs. Do not attach mcqs in json. Just generate the list [].
    }}"""
    mcq_prompt = PromptTemplate(template=template, input_variables=["context"])
    mcq_chain = load_qa_chain(mcq_model, chain_type="stuff", prompt=mcq_prompt)
    return mcq_chain

@app.post("/upload_data")
async def upload_data(file: bytes = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="Please upload a file.")
    
    text = file.decode("utf-8")  # assuming the file is a text file
    vector_store = get_vectorstore_from_text(text)
    vector_stores["text_data"] = vector_store
    return {"detail": "Text data uploaded successfully."}

@app.post("/prompt_config")
async def prompt_config(system_prompt: str = Form(...)):
    global llm_chain
    system_template = PromptTemplate(template=system_prompt, input_variables=["system_message"])
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    llm_chain = LLMChain(llm=model, prompt=system_template)
    return {"detail": "Prompt configuration updated successfully."}

@app.post("/user_query_prompt")
async def search_prompt(question: str = Form(...)):
    global llm_chain  # Access the global llm_chain variable

    if "text_data" not in vector_stores:
        raise HTTPException(status_code=400, detail="No vector stores available.")
    
    document_chunks = vector_stores["text_data"]["document_chunks"]
    embeddings = vector_stores["text_data"]["embeddings"]
    new_db = FAISS.from_documents(document_chunks, embeddings)
    docs = new_db.similarity_search(question)

    context = " ".join([doc.page_content for doc in docs])

    # Check if llm_chain is None (i.e., /prompt_config hasn't been called yet)
    if llm_chain is None:
        raise HTTPException(status_code=400, detail="Prompt configuration not set. Please call /prompt_config first.")

    # Use llm_chain to generate a response
    response = llm_chain({"input_documents": docs, "context": context, "question": question})

    # If the llm_chain cannot find an answer in the context, use the llm_chain again
    if response.get("output_text", "").lower().count("context") > 1:
        response = llm_chain({"question": question})

    # If the model is still not able to generate a response, provide a fallback response
    if not response.get("output_text") and not response.get("text"):
        response["output_text"] = "Sorry, I couldn't generate a response based on the provided context and question."

    return {"response": response.get("output_text", response.get("text", "No response generated."))}

@app.post("/upload_pdfs")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    file_ids = []
    for file in files:
        if file.filename.endswith(".pdf"):
            file_id = str(uuid.uuid4())
            with open(f"temp/{file_id}.pdf", "wb") as f:
                f.write(await file.read())
            file_ids.append(file_id)

    return {"file_ids": file_ids}

class MCQRequest(BaseModel):
    file_ids: List[str]
    context: str
    
@app.post("/generate_mcqs")
async def generate_mcqs(request: MCQRequest):
    text = ""
    for file_id in request.file_ids:
        with open(f"temp/{file_id}.pdf", "rb") as f:
            pdf_file = PyPDF2.PdfReader(f)
            for page_num in range(len(pdf_file.pages)):
                page_obj = pdf_file.pages[page_num]
                text += page_obj.extract_text()

    vector_store = get_vectorstore_from_text(text)
    new_db = FAISS.from_documents(vector_store["document_chunks"], vector_store["embeddings"])
    docs = new_db.similarity_search(f"Generate MCQ based on the {request.context}")
    mcq_chain = get_mcq_chain()
    mcq_response = mcq_chain({"input_documents": docs, "question": f"Generate MCQ based on the {request.context}"}, return_only_outputs=True)

    if mcq_response["output_text"].strip():
        try:
            cleaned_response = mcq_response["output_text"].strip().strip("```json").strip("```").strip()
            cleaned_response = cleaned_response.replace('```json', '').replace('```', '').strip()
            mcq_json = json.loads(cleaned_response)
            return mcq_json
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"The response is not a valid JSON string. Error: {str(e)}")
    else:
        raise HTTPException(status_code=400, detail="The response is empty.")