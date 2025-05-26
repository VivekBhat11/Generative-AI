import wikipedia
import cohere
from langchain.llms import Cohere as CohereLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.embeddings.base import Embeddings
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser

# STEP 1: Initialize Cohere API
COHERE_API_KEY = "D9qGIfhKOs3xfBmAx7alQalXoPfoJVUFSPBsDCr5"  # Replace with your key
co = cohere.Client(COHERE_API_KEY)
llm = CohereLLM(cohere_api_key=COHERE_API_KEY, model="command-r-plus")

# Custom Embeddings
class CustomCohereEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return co.embed(texts=texts, model="embed-english-v3.0", input_type="search_document").embeddings
    def embed_query(self, text):
        return self.embed_documents([text])[0]

embedding = CustomCohereEmbeddings()

# STEP 2: Load and summarize a text file
print("\n📄 Document Summary")
loader = TextLoader("skill_india.txt")  # Upload any .txt file here
docs = loader.load()
prompt = PromptTemplate.from_template("Give a summary of the following:\n\n{context}")
chain = LLMChain(llm=llm, prompt=prompt)
summary = chain.run(context=docs[0].page_content)
print("\nSummary:\n", summary)

# STEP 3: Extract Institution Info
print("\n🏛 Institution Info from Wikipedia")
class InstitutionDetails(BaseModel):
    founder: str = Field(..., description="Founder")
    founded: str = Field(..., description="Founded Year")
    branches: str = Field(..., description="Branches")
    employees: str = Field(..., description="Employees")
    summary: str = Field(..., description="4-line summary")

parser = PydanticOutputParser(pydantic_object=InstitutionDetails)
inst_name = input("Enter Institution Name: ")
try:
    content = wikipedia.page(inst_name).content[:2000]
    inst_prompt = PromptTemplate(
        input_variables=["context", "institution"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
        template="""
Use the following Wikipedia content to answer:

{context}

{format_instructions}

Institution name: {institution}
"""
    )
    chain = LLMChain(llm=llm, prompt=inst_prompt)
    result = chain.run(context=content, institution=inst_name)
    data = parser.parse(result)
    print("\nFounder:", data.founder)
    print("Founded:", data.founded)
    print("Branches:", data.branches)
    print("Employees:", data.employees)
    print("Summary:", data.summary)
except Exception as e:
    print("Error fetching info:", e)

# STEP 4: IPC Chatbot
print("\n📘 Indian Penal Code Chatbot")
loader = TextLoader("ipc.txt")  # Save Indian Penal Code in a .txt file
docs = loader.load()
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)
vectorstore = FAISS.from_documents(chunks, embedding)
retriever = vectorstore.as_retriever()
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

while True:
    q = input("\nAsk about IPC (or type 'exit'): ")
    if q.lower() == 'exit':
        break
    print("Answer:", qa.run(q))
