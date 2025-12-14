Financial RAG System Design Report
1. Executive Summary
This report outlines the architecture of a Retrieval-Augmented Generation (RAG) system designed to answer factual questions from financial 10-K documents (Apple & Tesla). The system prioritizes high precision, strict citation tracking, and robust handling of tabular data.
2. Ingestion & Chunking Strategy
Effective chunking is critical for financial documents where context (like headers) often spans multiple paragraphs.
•	Text Processing:
o	Method: RecursiveCharacterTextSplitter
o	Parameters: Chunk Size: 1000 characters, Overlap: 200 characters.
o	Rationale: A larger chunk size of 1000 ensures that complete sentences and their immediate context are preserved. The 200-character overlap prevents information loss at the boundaries of splits, ensuring that references (e.g., "The company...") remain linked to their subjects.
•	Table Processing (Critical Feature):
o	Method: Tables are extracted whole using pdfplumber and converted to Markdown format.
o	Rationale: Financial data relies on row-column alignment. Standard text splitters destroy this structure. By treating tables as standalone atomic chunks, we ensure the LLM receives the full financial context (Year vs. Revenue) without fragmentation.
3. Retrieval Strategy
A "Hybrid + Rerank" pipeline was chosen to balance semantic understanding with keyword precision.
1.	Vector Search (ChromaDB): Captures semantic meaning (e.g., "earnings" ≈ "revenue").
2.	Keyword Search (BM25): captures exact matches for unique identifiers (e.g., "6.07%", "September 28, 2024").
3.	Re-ranking (Cross-Encoder): The top 10 results from both streams are fused and re-scored using ms-marco-MiniLM-L-6-v2. This ensures that the context fed to the LLM is highly relevant, reducing hallucinations.
4. LLM Selection: Llama 3 vs. Phi-3
The system was initially tested with Microsoft/Phi-3-mini but migrated to Meta/Llama-3.
•	Phi-3-Mini (3.8B) Limitations: While efficient, Phi-3 struggled with the complex "negative constraint" instructions. It often attempted to answer out-of-scope questions or hallucinated page numbers despite explicit instructions not to. It also struggled to consistently format citations (e.g., [Ref: Source, Page]).
•	Llama-3 (8B) Advantages: Llama-3 demonstrates significantly stronger instruction-following capabilities.
o	Reasoning: It can better correlate data across complex Markdown tables.
o	Compliance: It strictly adheres to negative constraints (refusing to answer when data is missing).
o	Formatting: It reliably generates the regex-parseable citation format required for the source tracking feature.
5. Out-of-Scope (OOS) Handling
Preventing hallucinations is a strict requirement for financial analysis. OOS handling is implemented via a three-layer defence:
1.	Prompt Engineering (Negative Constraints): The system prompt explicitly instructs the model: "If the answer is not in the text, respond EXACTLY: 'This question cannot be answered based on the provided documents.'"
2.	Citation Validation: The post-processing logic parses the answer for [Ref: ...] tags. If the model provides an answer but fails to cite a source from the provided context, the answer is flagged or discarded (depending on configuration).
3.	Source Filtering: The specific string "This question cannot be answered..." is hardcoded as a stop condition. If detected, the sources list is forcibly cleared to prevent misleading attributions.

