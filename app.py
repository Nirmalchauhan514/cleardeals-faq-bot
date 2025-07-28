import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load the CSV file
faq_data = pd.read_csv("faq_data.csv")

# Clean data
faq_data = faq_data.dropna(subset=["Question"])
faq_data['Question'] = faq_data['Question'].astype(str)

# Load sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Embed the questions
question_embeddings = [model.encode(q) for q in faq_data['Question']]

# App title
st.title("ClearDeals FAQ Chatbot")

# Show all available questions

# Input box
with st.expander("📋 View All FAQs"):
    selected_question = st.selectbox("Choose a question:", faq_data['Question'])
    if st.button("Ask Selected Question"):
        user_input = selected_question
    else:
        user_input = st.text_input("🔍 Or ask your own question:")

if user_input:
    user_embedding = model.encode(user_input)
    similarities = [cosine_similarity([qe], [user_embedding])[0][0] for qe in question_embeddings]
    most_similar_idx = similarities.index(max(similarities))
    
    st.markdown("### ✅ Answer:")
    st.write(faq_data.iloc[most_similar_idx]['Answer'])
