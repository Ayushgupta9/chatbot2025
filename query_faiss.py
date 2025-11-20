import re
from langdetect import detect
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# 1. Load FAISS index
def load_faiss_index(index_path):
    try:
        return faiss.read_index(index_path)
    except Exception as e:
        raise FileNotFoundError(f"Error loading FAISS index: {e}")


# 2. Perform query search in FAISS index
def search_faiss_index(query, index, model, top_k=1):
    try:
        query_embedding = model.encode([query], convert_to_tensor=False)
        query_embedding = np.array(query_embedding).astype('float32')
        distances, indices = index.search(query_embedding, top_k)
        return distances, indices
    except Exception as e:
        raise RuntimeError(f"Error performing FAISS search: {e}")


# 3. Load chunks from file
def load_chunks_from_file(file_path='chunks.txt'):
    try:
        with open(file_path, 'r') as f:
            return [line.strip() for line in f.readlines()]
    except Exception as e:
        raise FileNotFoundError(f"Error loading chunks file: {e}")


# 4. Language-aware search
def language_aware_search(query, model, index, chunks, top_k=1):
    lang = detect(query)
    print(f"Detected language: {lang}")

    if lang not in ['de', 'en']:
        return f"Sorry, the language '{lang}' is not supported. Please use English or German."

    distances, indices = search_faiss_index(query, index, model, top_k)
    relevant_chunks = [chunks[i] for i in indices[0] if detect(chunks[i]) == lang]

    if not relevant_chunks:
        return f"No relevant matches found in {lang}."

    return generate_human_like_response(relevant_chunks, query)


# 5. Generate Human-like Response
def generate_human_like_response(chunks, query):
    """
    Refine and synthesize a human-like response based on retrieved chunks.
    """
    # Example of customizing response synthesis
    response = []
    for chunk in chunks:
        if 'medicine' in query.lower() or 'drug' in query.lower():
            match = re.search(r'\b(medicine|drug):\s*([\w\s]+)', chunk, re.IGNORECASE)
            if match:
                response.append(f"A suggested medicine is: {match.group(2)}.")
        elif 'treatment' in query.lower():
            match = re.search(r'\btreatment:\s*([\w\s,]+)', chunk, re.IGNORECASE)
            if match:
                response.append(f"A recommended treatment is: {match.group(1)}.")
        else:
            response.append(concise_output(chunk))

    # Combine responses for a final output
    return "\n".join(response) if response else concise_output(chunks[0])


# 6. Concise Output
def concise_output(text, max_length=240):
    """
    Truncate text to a maximum length for concise responses.
    """
    return text[:max_length] + "..." if len(text) > max_length else text


# 7. Main Query Mechanism
def main_query(index_path, query, chunks_file_path='chunks.txt', model_name='paraphrase-multilingual-MiniLM-L12-v2', top_k=1):
    try:
        # Load FAISS index
        index = load_faiss_index(index_path)
        print("FAISS index loaded successfully.")

        # Load SentenceTransformer model
        model = SentenceTransformer(model_name)
        print(f"Model '{model_name}' loaded successfully.")

        # Load preprocessed chunks
        chunks = load_chunks_from_file(chunks_file_path)
        print(f"Loaded {len(chunks)} chunks from file.")

        # Perform search and synthesize response
        result = language_aware_search(query, model, index, chunks, top_k)
        print("\nResult:\n", result)
    except Exception as e:
        print(f"Error: {e}")


# Example Usage
if __name__ == "__main__":
    index_path = 'faiss_index'  # Path to FAISS index
    chunks_file_path = 'chunks.txt'  # Path to chunks file
    query = input("Enter your query: ")  # Dynamic user input
    main_query(index_path, query, chunks_file_path)