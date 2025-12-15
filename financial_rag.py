import os
import json
import pdfplumber
import numpy as np
from typing import List, Dict

# --- LangChain & Vector Store Imports ---
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder
import re

# --- CONFIGURATION ---
# UPDATE THESE PATHS TO MATCH YOUR FILE LOCATION EXACTLY
FILES = {
    "financial_docs/10-Q4-2024-As-Filed.pdf": "Apple10-K",
    "financial_docs/tsla-20231231-gen.pdf": "Tesla10-K"
}
DB_PATH = "./chroma_db_final"
LLM_MODEL = "llama3:latest" # Ensure you have run 'ollama pull llama3'

# --- PART 1: INGESTION (Table-Aware) ---
def extract_pdf_data(files_dict):
    """Extracts text and tables (as Markdown) from PDFs."""
    documents = []
    print("--- Starting Extraction ---")
    
    for path, source_name in files_dict.items():
        if not os.path.exists(path):
            print(f"WARNING: File {path} not found. Skipping.")
            continue
            
        print(f"Processing {source_name}...")
        try:
            with pdfplumber.open(path) as pdf:
                for i, page in enumerate(pdf.pages):
                    # 1. Extract Tables
                    tables = page.extract_tables()
                    if tables:
                        for table in tables:
                            # Clean table and convert to Markdown
                            clean_table = [[str(cell) if cell else "" for cell in row] for row in table]
                            if len(clean_table) > 0:
                                header = "| " + " | ".join(clean_table[0]) + " |"
                                separator = "| " + " | ".join(["---"] * len(clean_table[0])) + " |"
                                body = "\n".join(["| " + " | ".join(row) + " |" for row in clean_table[1:]])
                                markdown_table = f"\n{header}\n{separator}\n{body}\n"
                                
                                documents.append(Document(
                                    page_content=markdown_table,
                                    metadata={"source_name": source_name, "page": i + 1, "type": "table"}
                                ))

                    # 2. Extract Text
                    text = page.extract_text()
                    if text:
                        documents.append(Document(
                            page_content=text,
                            metadata={"source_name": source_name, "page": i + 1, "type": "text"}
                        ))
        except Exception as e:
            print(f"Error reading {path}: {e}")
            
    return documents

def setup_database():
    """Ingests data if DB doesn't exist, otherwise loads it."""
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    
    # Check if DB exists AND has data
    if os.path.exists(DB_PATH) and os.listdir(DB_PATH):
        print("Checking existing Vector Database...")
        vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embedding_model, collection_name="financial_rag")
        
        # FIX: Check if collection is actually empty to avoid BM25 crash
        if len(vectorstore.get()['ids']) > 0:
            print("Database loaded successfully.")
            return vectorstore
        else:
            print("Database folder exists but is empty. Re-ingesting...")
    
    # Start Fresh Ingestion
    print("Creating new Database...")
    raw_docs = extract_pdf_data(FILES)
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_docs = []
    for doc in raw_docs:
        if doc.metadata["type"] == "table":
            final_docs.append(doc) # Keep tables intact
        else:
            final_docs.extend(splitter.split_documents([doc]))
            
    vectorstore = Chroma.from_documents(
        documents=final_docs,
        embedding=embedding_model,
        persist_directory=DB_PATH,
        collection_name="financial_rag"
    )
    print(f"Ingested {len(final_docs)} chunks.")
    return vectorstore

# --- PART 2: RETRIEVAL SETUP ---
print("Initializing System...")
vectorstore = setup_database()
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Build In-Memory BM25 Index
print("Building Keyword Index (BM25)...")
all_docs_data = vectorstore.get() 
bm25_docs = [
    Document(page_content=txt, metadata=meta) 
    for txt, meta in zip(all_docs_data["documents"], all_docs_data["metadatas"])
]
bm25_retriever = BM25Retriever.from_documents(bm25_docs)
bm25_retriever.k = 10

def manual_hybrid_search(query, top_k=5):
    """Combines Vector + Keyword Search + Re-ranking"""
    # 1. Fetch Candidates
    vector_results = vectorstore.similarity_search(query, k=10)
    keyword_results = bm25_retriever.invoke(query)
    
    # 2. Deduplicate
    combined = {doc.page_content: doc for doc in vector_results + keyword_results}
    unique_docs = list(combined.values())
    
    if not unique_docs: return []

    # 3. Re-rank
    pairs = [[query, doc.page_content] for doc in unique_docs]
    scores = reranker.predict(pairs)
    
    # Sort descending
    sorted_idxs = np.argsort(scores)[::-1]
    final_docs = [unique_docs[i] for i in sorted_idxs[:top_k]]
    
    return final_docs


# --- PART 3: GENERATION ---
llm = ChatOllama(model=LLM_MODEL, temperature=0)

system_prompt = """You are a strictly factual financial analyst assistant.
Answer the question based ONLY on the provided context.

RULES:
1. **Citation Required:** When you find the answer, you MUST append a reference tag at the end of the sentence.
   Format: [Ref: {{Source Name}}, Page {{Page Number}}]
   Example: Revenue was $50 billion [Ref: Apple10-K, Page 32].
2. **Tables:** Scan Markdown tables carefully for exact numbers.
3. **Missing Info:** If the answer is not in the text, respond EXACTLY: "This question cannot be answered based on the provided documents."
4. **Out of Scope:** If asking about unrelated topics, respond EXACTLY: "This question cannot be answered based on the provided documents."
5. **Format:** Provide ONLY the direct answer (number, date, or sentence). Do NOT include [Source, Page] citations in your text output.


Context:
{context}
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", "{question}")
])

chain = prompt_template | llm | StrOutputParser()

def answer_question(question):
    print(f"Thinking on: '{question}'...")

    # 1. Retrieve Candidate Docs (we still get Top 5 to search through)
    docs = manual_hybrid_search(question, top_k=5)

    # 2. Format Context with clear headers for the LLM to read
    context_str = ""
    for doc in docs:
        src = doc.metadata.get("source_name", "Unknown")
        pg = doc.metadata.get("page", "Unknown")
        # The LLM sees this header and uses it to create the [Ref: ...] tag
        context_str += f"\n--- Source Name: {src}, Page Number: {pg} ---\n{doc.page_content}\n"

    # 3. Generate Answer
    raw_answer = chain.invoke({"context": context_str, "question": question})

    # 4. Extract Citations using Regex
    # We look for patterns like [Ref: Apple10-K, Page 32]
    pattern = r"\[Ref: (.*?), Page (\d+)\]"
    matches = re.findall(pattern, raw_answer)

    unique_sources = set()
    for match in matches:
        source_name, page_num = match
        unique_sources.add(f"{source_name}, Page {page_num}")

    sources_list = list(unique_sources)

    # 5. Clean the Answer
    # Remove the [Ref: ...] tags from the text so the final answer is clean
    clean_answer = re.sub(pattern, "", raw_answer).strip()

    # Fallback: If LLM gave an answer but forgot to tag it, warn the user (or leave empty)
    if not sources_list and "cannot be answered" not in clean_answer:
        # Optional: You could fallback to listing top 1 doc, but for accuracy, empty is honest.
        pass

    return clean_answer, sources_list

# --- PART 4: EXECUTION ---
questions = [
      {"question_id": 1, "question": "What was Apples total revenue for the fiscal year ended September 28, 2024?"},
        {"question_id": 2, "question": "How many shares of common stock were issued and outstanding as of October 18, 2024?"},
        {"question_id": 3, "question": "What is the total amount of term debt (current + non-current) reported by Apple as of September 28, 2024?"},
        {"question_id": 4, "question": "On what date was Apples 10-K report for 2024 signed and filed with the SEC?"},
        {"question_id": 5, "question": "Does Apple have any unresolved staff comments from the SEC as of this filing? How do you know?"},
        {"question_id": 6, "question": "What was Teslas total revenue for the year ended December 31, 2023?"},
        {"question_id": 7, "question": "What percentage of Teslas total revenue in 2023 came from Automotive Sales (excluding Leasing)?"},
        {"question_id": 8, "question": "What is the primary reason Tesla states for being highly dependent on Elon Musk?"},
        {"question_id": 9, "question": "What types of vehicles does Tesla currently produce and deliver?"},
        {"question_id": 10, "question": "What is the purpose of Teslas 'lease pass-through fund arrangements'?"},
        {"question_id": 11, "question": "What is Teslas stock price forecast for 2025?"},
        {"question_id": 12, "question": "Who is the CFO of Apple as of 2025?"},
        {"question_id": 13, "question": "What color is Teslas headquarters painted?"}
]

final_output = []

print("\n--- Starting RAG Loop ---")
for q in questions:
    ans, srcs = answer_question(q["question"])
    result_entry = {
        "question_id": q["question_id"],
        "answer": ans,
        "sources": srcs
    }
    final_output.append(result_entry)
    print(json.dumps(result_entry, indent=4))

output_filename = "financial_rag_answers.json"
with open(output_filename, "w") as f:
    json.dump(final_output, f, indent=4)

print(final_output)
