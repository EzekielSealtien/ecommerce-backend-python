from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
import os
from dotenv import load_dotenv


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Nom EXACT de mon index Pinecone
INDEX_NAME = "ecommerceindex2"
TEXT_KEY = "text"

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
)

vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    text_key=TEXT_KEY
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.2,
    timeout=60,
    max_retries=3,
)

PROMPT_TEMPLATE = """
Tu es un assistant e-commerce. Réponds UNIQUEMENT à partir du CONTEXTE.
Si l'information n'est pas dans le contexte, dis exactement :
"Je n’ai pas cette information dans la base de connaissance."

CONSIGNE:
- Réponse claire et structurée
- Si pertinent, utilise des puces
- Ne devine pas

CONTEXTE:
{context}

QUESTION:
{input}

RÉPONSE:
""".strip()

prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

combine_docs_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
rag_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=combine_docs_chain)



def ask_ecommerce_chatbot(user_input: str) -> str:
    """
    Prend la question utilisateur et renvoie la réponse générée par le RAG.
    """
    result = rag_chain.invoke({"input": user_input})
    return result["answer"]




