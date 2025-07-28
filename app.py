import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

# Load sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the CSV file
faq_data = pd.read_csv("faq_data.csv")

# Embed the questions and ensure they're stored as proper lists
faq_data['embedding'] = faq_data['Question'].apply(lambda x: model.encode(str(x)).tolist())

# Drop rows with invalid embeddings
faq_data = faq_data.dropna(subset=['embedding'])

# Streamlit UI
st.title("ClearDeals FAQ Chatbot")
user_input = st.text_input("Ask a question:")

if user_input:
    # Encode user input
    user_embedding = model.encode(user_input)

    def safe_similarity(row_embedding):
        # Ensure the embedding is a list and has the correct shape
        if isinstance(row_embedding, list) and len(row_embedding) == len(user_embedding):
            return cosine_similarity([row_embedding], [user_embedding])[0][0]
        else:
            return -1  # Return very low similarity for invalid rows

    # Calculate similarity safely
    faq_data['similarity'] = faq_data['embedding'].apply(safe_similarity)

    # Get the most similar answer
    best_match_idx = faq_data['similarity'].idxmax()

    if faq_data['similarity'][best_match_idx] < 0.5:
        st.write("Sorry, I couldn't find a good match. Please try rephrasing.")
    else:
        st.write("Answer:")
        st.write(faq_data.loc[best_match_idx, 'Answer'])
