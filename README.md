# RAG_financial_doc_with_open_source_LLM
Built a Retrieval-Augmented Generation (RAG) system that answers complex financial and legal questions by retrieving information from Apple’s 2024 10-K and Tesla’s 2023 10-K, real SEC filings containing financial statements, legal proceedings etc. details and generating accurate, well-sourced responses using open source LLM Llama3.
Collab link: https://colab.research.google.com/drive/1F8D4oE2mzBQAXjL6YzLk3qeTyCGTaEGy?usp=sharing

# Financial RAG Analyst

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-Integration-orange?style=for-the-badge&logo=langchain&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-Llama3-black?style=for-the-badge&logo=ollama&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-green?style=for-the-badge)

A specialized Retrieval-Augmented Generation (RAG) system designed for high-precision financial analysis of 10-K filings. This project solves common RAG pitfalls—like hallucination and table fragmentation—by using a **hybrid search pipeline** and **table-aware ingestion**.

## Key Features

* **Table-Aware Ingestion**: Unlike standard text splitters that scramble data, this system extracts tables as standalone Markdown chunks, preserving row/column relationships for accurate financial lookup.
* **Hybrid Search**: Combines **BM25 (Keyword)** for exact numbers with **Vector Search (Semantic)** for conceptual understanding.
* **Re-Ranking**: Uses `Cross-Encoder/MS-Marco` to re-score and filter the top retrieved documents, ensuring only the most relevant context reaches the LLM.
* **Strict Citations**: The model is engineered to provide precise source tracking (e.g., `[Ref: Apple10-K, Page 32]`) and strictly refuses out-of-scope questions.

## Architecture


The pipeline follows these steps:
1.  **Ingest**: PDFs (Apple/Tesla 10-Ks) are processed; text is chunked, tables are converted to Markdown.
2.  **Store**: Chunks are embedded using `all-mpnet-base-v2` and stored in ChromaDB.
3.  **Retrieve**: User queries trigger parallel Vector and BM25 searches.
4.  **Rank**: Top results are re-ranked by relevance using a Cross-Encoder.
5.  **Generate**: Llama-3 answers the query using strictly the provided context, appending citations.

## Project Structure

```bash
├── financial_docs/          # Directory for PDF inputs (e.g., Apple-10K.pdf)
├── chroma_db_final/         # Persistent Vector Database (generated)
├── rag_with_sources.py      # Main application logic
```

## Quick Start:
### Option 1: Google Colab / Kaggle (Recommended)
This project is optimized to run in free cloud environments with GPU support.Open the notebook.Change Runtime to T4 GPU.Run the setup cells to install Ollama and LangChain.Execute the pipeline.
### Option 2: Local Installation
Prerequisites:Python 3.10+Ollama installed and running (ollama serve)
1. Clone the Repository
(Bash) git clone [https://github.com/Ambachat/RAG_financial_doc_with_open_source_LLM.git](https://github.com/yourusername/RAG_financial_doc_with_open_source_LLM.git)
cd financial-rag-analyst
2. Install Dependencies
pip install langchain-community langchain-chroma langchain-huggingface \
pdfplumber sentence-transformers chromadb
3. Pull the Model
(Bash) ollama pull llama3
4. Run the Analyst
(Bash) python financial_rag.py

#### Configuration:
You can tweak the following constants in financial_rag.py to optimize performance:
Parameter | Default | Description
--- | --- | ---
CHUNK_SIZE|1000|Size of text chunks (characters)
CHUNK_OVERLAP | 200 | Overlap between chunks to preserve context.
TOP_K | 5 | Number of documents to retrieve per query.
LLM_MODEL | llama3 | The Ollama model to use (switchable to phi3).

#### Example Output:
Question: "What was Apple's total revenue for 2024?"
Answer: Apple's total net sales for the fiscal year 2024 were $391.0 billion.Sources: ['Apple10-K, Page 32']

Question: "Who won the 2024 Super Bowl?"
Answer: This question cannot be answered based on the provided documents.Sources: []

#### Contributing:
Contributions are welcome! Please read design_report.md to understand the chunking strategy before submitting Pull Requests.

#### License: 
This project is licensed under the MIT License.

├── design_report.md          # Detailed architectural decision record
└── README.md                # This file
