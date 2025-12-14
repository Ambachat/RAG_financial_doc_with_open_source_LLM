Financial RAG System Design Report

1. Executive Summary

This report describes a Retrieval-Augmented Generation (RAG) system built to answer factual questions from financial 10-K documents, specifically for Apple and Tesla. The system is designed with a strong focus on accuracy, proper source citations, and reliable handling of financial tables.

2. Ingestion and Chunking Strategy

Proper chunking is very important for financial documents because important context often spans multiple paragraphs.

For normal text,  a recursive text splitting approach is used . Each chunk is around 1000 characters long, with an overlap of 200 characters between chunks. This helps preserve full sentences and ensures that references like “the company” still make sense across chunk boundaries.

Handling tables is treated as a critical requirement for financial docs since it contain important numerical values. Financial tables are extracted as a whole using pdfplumber and converted into Markdown format. Tables are never split into smaller chunks. This is done because financial meaning depends heavily on row and column alignment (for example, year versus revenue). Keeping tables intact ensures that the language model receives complete and accurate financial context.

3. Retrieval Strategy

The system uses a hybrid retrieval approach combined with re-ranking to ensure high-quality results.

The documents are embedded using sentence-transformers/all-mpnet-base-v2 model  and  stored in chroma db.

Vector search using ChromaDB captures semantic meaning, allowing the system to understand related concepts such as “earnings” and “revenue.” In parallel, keyword search using BM25 ensures exact matches for specific values like percentages or dates.

The top results from both retrieval methods are combined and re-ranked using a cross-encoder model (ms-marco-MiniLM-L-6-v2). This step ensures that only the most relevant and precise information is passed to the language model, which significantly reduces hallucinations.

4. LLM Selection:  Llama-3


Llama-3 performs much better in these areas. It follows instructions more reliably, reasons more accurately over complex Markdown tables, and strictly respects negative constraints. It also consistently generates citations in a fixed, machine-readable format, which is essential for source tracking.

It  also  provides good citation for the  answers.

5. Out-of-Scope Question Handling

Preventing hallucinations is a strict requirement for financial analysis. The system uses three layers of protection.

First, the prompt clearly instructs the model to respond with a fixed sentence if the answer is not present in the documents. Second, the system validates citations in the final answer. If an answer is provided without valid references, it is flagged or discarded. Third, if the predefined out-of-scope response is detected, all sources are automatically removed to avoid misleading attribution.
