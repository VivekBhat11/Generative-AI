import streamlit as st
import wikipedia
import cohere
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain.llms import Cohere as CohereLLM
from langchain.embeddings.base import Embeddings
from langchain.chains import RetrievalQA, LLMChain
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate

# ========== SETUP ==========
st.set_page_config(page_title="Generative AI App", layout="centered")
st.title("🤖 Generative AI Project – Skill India, Institutions & IPC")

COHERE_API_KEY = "D9qGIfhKOs3xfBmAx7alQalXoPfoJVUFSPBsDCr5"
co = cohere.Client(COHERE_API_KEY)

# Custom Embeddings Class
class CustomCohereEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return co.embed(texts=texts, model="embed-english-v3.0", input_type="search_document").embeddings
    def embed_query(self, text):
        return self.embed_documents([text])[0]

embedding = CustomCohereEmbeddings()
llm = CohereLLM(cohere_api_key=COHERE_API_KEY, model="command-r-plus")

# ========== 1. SKILL INDIA ==========
st.header("📘 Ask About Virat Kohli")
try:
    loader = TextLoader("skill_india.txt")
    documents = loader.load()
    chunks = CharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(documents)
    vectorstore = FAISS.from_documents(chunks, embedding)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever(), chain_type="stuff")

    question = st.text_input("Type your Skill India question:")
    if question:
        answer = qa.run(question)
        st.success(f"🧠 Answer: {answer}")
except FileNotFoundError:
    st.warning("skill_india.txt not found. Please upload the file.")

# ========== 2. INSTITUTION INFO ==========
st.header("🏛 Institution Information (From Wikipedia)")

class InstitutionDetails(BaseModel):
    founder: str = Field(..., description="Founder of the Institution")
    founded: str = Field(..., description="Year it was founded")
    branches: str = Field(..., description="Current branches")
    employees: str = Field(..., description="Number of employees")
    summary: str = Field(..., description="Brief 4-line summary")

parser = PydanticOutputParser(pydantic_object=InstitutionDetails)

prompt = PromptTemplate(
    template="""
Use the Wikipedia content below to extract the following details:
{format_instructions}

Institution: {institution}
Wikipedia content:
{context}
""",
    input_variables=["context", "institution"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

institution_name = st.text_input("Enter Institution Name:")
if st.button("Fetch Info"):
    try:
        wiki_text = wikipedia.page(institution_name).content[:2000]
        chain = LLMChain(llm=llm, prompt=prompt)
        output = chain.run({"context": wiki_text, "institution": institution_name})
        data = parser.parse(output)
        st.write(f"**Founder:** {data.founder}")
        st.write(f"**Founded:** {data.founded}")
        st.write(f"**Branches:** {data.branches}")
        st.write(f"**Employees:** {data.employees}")
        st.write(f"**Summary:** {data.summary}")
    except:
        st.error("Institution not found or error occurred.")

# ========== 3. IPC Chatbot ==========
st.header("📜 Indian Penal Code Chatbot")
try:
    loader = TextLoader("ipc.txt")
    documents = loader.load()
    chunks = CharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(documents)
    ipc_vectorstore = FAISS.from_documents(chunks, embedding)
    ipc_qa = RetrievalQA.from_chain_type(llm=llm, retriever=ipc_vectorstore.as_retriever(), chain_type="stuff")

    ipc_question = st.text_input("Ask a question about the IPC:")
    if ipc_question:
        ipc_answer = ipc_qa.run(ipc_question)
        st.success(f"📘 IPC Answer: {ipc_answer}")
except FileNotFoundError:
    st.warning("ipc.txt not found. Please upload the file.")
