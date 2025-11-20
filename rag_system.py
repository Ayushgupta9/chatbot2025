import os
import pdfplumber
import re
from nltk.tokenize import sent_tokenize
from langdetect import detect
from sentence_transformers import SentenceTransformer, util
import numpy as np
import faiss
from transformers import pipeline

# 1. Extract text from PDFs
def extract_text_from_pdfs(pdf_folder):
    text = ""
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
    return text

# 2. Clean text
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespaces
    text = re.sub(r'[^\w\s,.!?-]', '', text)  # Remove special characters
    return text

# 3. Filter sentences by language
def filter_language(sentences, allowed_languages=('en', 'de')):
    filtered = []
    for sentence in sentences:
        try:
            # Ensure the sentence has enough content to detect the language
            if len(sentence.strip()) > 20 and detect(sentence) in allowed_languages:
                filtered.append(sentence)
        except Exception as e:
            # Skip sentences that cause exceptions
            continue
    return filtered

# 4. Chunk text
def chunk_text(sentences, chunk_size=300, overlap=50):
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)
        if current_length + sentence_length > chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-overlap:]  # Overlap with previous chunk
            current_length = sum(len(s) for s in current_chunk)
        current_chunk.append(sentence)
        current_length += sentence_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# 5. Embed chunks
def create_embeddings(chunks, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, convert_to_tensor=False, show_progress_bar=True)
    return np.array(embeddings), model

# 6. Save embeddings to FAISS
def save_faiss_index(embeddings, index_path='faiss_index'):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, index_path)

# 7. Load text generator (using Hugging Face API)
def load_text_generator(model_name="distilgpt2", token=None):
    if token:
        return pipeline("text-generation", model=model_name, tokenizer=model_name, use_auth_token=token)
    return pipeline("text-generation", model=model_name)

# 8. Refined retrieval and response synthesis
def retrieve_and_generate_response(query, index_path, chunks, model, generator, top_n=3):
    # Load the FAISS index
    index = faiss.read_index(index_path)
    
    # Encode the query
    query_embedding = model.encode(query, convert_to_tensor=False)
    
    # Retrieve top N relevant chunks
    _, indices = index.search(np.array([query_embedding]), top_n)
    
    # Combine relevant chunks
    relevant_chunks = [chunks[i] for i in indices[0]]
    combined_content = " ".join(relevant_chunks)

    # Use the generator to synthesize a human-like response
    prompt = f"Based on the following content, provide a human-like response to the query:\n\nContent: {combined_content}\n\nQuery: {query}\n\nResponse:"
    response = generator(prompt, max_length=250, num_return_sequences=1)[0]['generated_text']
    
    return response

# 9. Main Process
def main(pdf_folder, index_path, query, hf_token=None):
    # Step 1: Extract text
    raw_text = extract_text_from_pdfs(pdf_folder)
    print("Text extracted from PDFs.")

    # Step 2: Clean text
    cleaned_text = clean_text(raw_text)
    print("Text cleaned.")

    # Step 3: Tokenize into sentences
    sentences = sent_tokenize(cleaned_text)
    print(f"{len(sentences)} sentences tokenized.")

    # Step 4: Filter by language
    filtered_sentences = filter_language(sentences)
    print(f"{len(filtered_sentences)} sentences after language filtering.")

    # Step 5: Chunk text
    chunks = chunk_text(filtered_sentences, chunk_size=300)  # Reduced chunk size
    print(f"{len(chunks)} chunks created.")

    # Step 6: Create embeddings
    embeddings, model = create_embeddings(chunks)
    print("Embeddings created.")

    # Step 7: Save to FAISS
    save_faiss_index(embeddings, index_path)
    print(f"FAISS index saved to {index_path}.")

    # Step 8: Load text generator
    generator = load_text_generator(token=hf_token)
    print("Text generation model loaded.")

    # Step 9: Retrieve and synthesize response
    response = retrieve_and_generate_response(query, index_path, chunks, model, generator)
    print("\nRefined Response:\n", response)

    # Cleanup to free up memory
    del embeddings
    del model
    del generator

# Specify the folder where your PDFs are stored, FAISS index path, query, and token
pdf_folder = '/Users/ayushgupta/Desktop/ai_lab'
index_path = 'faiss_index'
query = "Explain the main concept from the document."
hf_token = "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"  # Add your token here

if __name__ == "__main__":
    main(pdf_folder, index_path, query, hf_token)