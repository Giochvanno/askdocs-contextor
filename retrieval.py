"""
retrieval.py — lightweight retrieval of relevant chunks without a vector DB.
 
Logic:
- documents are split into chunks;
- if everything fits the budget, we send all of it;
- otherwise BM25 selects the chunks most relevant to the question.
 
This is a "mini-RAG": it scales to large / multiple documents,
but without embeddings or external dependencies.
"""