from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from rag import ask_ecommerce_chatbot
import os


class UserInput(BaseModel):
	question: str

app = FastAPI()

# Allow cross-origin requests from dev frontend
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

@app.get("/")
def test():
    return {"message": "E-commerce Chatbot API is running. Thank you Jesus Christ!"}

@app.post("/api/chatbot-response")
def chatbot_response(req: UserInput):
	if ask_ecommerce_chatbot is None:
		raise HTTPException(status_code=500, detail="RAG module unavailable (import failed)")
	try:
		answer = ask_ecommerce_chatbot(req.question)
		return {"answer": answer}
	except Exception as e:
		raise HTTPException(status_code=500, detail="Une Erreur est survenue lors du traitement de la demande.")



