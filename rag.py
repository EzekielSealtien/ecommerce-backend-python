import os
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

load_dotenv()

a=1
loader = TextLoader("informations.txt")
text_documents = loader.load()

#Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=150)
documents = text_splitter.split_documents(text_documents)


def ask_ecommerce_chatbot(question):
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


    model = ChatOpenAI(model="gpt-4", temperature=0.0,openai_api_key=OPENAI_API_KEY)
    parser = StrOutputParser()
    template = """
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
    """

    prompt = ChatPromptTemplate.from_template(template)


    # Generate the embeddings for an arbitrary query
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)

    #use pinecone as a vector store
    pinecone_index_name = "project"


    pinecone = PineconeVectorStore.from_documents(
    documents=documents, embedding=embeddings, index_name=pinecone_index_name
    )

    chain = (
    {"context": pinecone.as_retriever(), "input": RunnablePassthrough()}
    | prompt
    | model
    | parser
    )
    response=chain.invoke(question)
    return response

