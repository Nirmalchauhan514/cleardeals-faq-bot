import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load the CSV file
faq_data = pd.read_csv("faq_data.csv")

# Load sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Embed the questions
faq_data['embedding'] = faq_data['Question'].apply(lambda x: model.encode(x))

# App title
st.title("ClearDeals FAQ Chatbot")

# Show all available questions
with st.expander("📋 View All FAQs"):
    for i, row in faq_data.iterrows():
        st.markdown(f"**Q{i+1}. {row['Question']}**")

# Input box
user_input = st.text_input("🔍 Ask a question:")

# Answer logic
if user_input:
    user_embedding = model.encode(user_input)
    similarities = faq_data['embedding'].apply(lambda x: cosine_similarity([x], [user_embedding])[0][0])
    most_similar_idx = similarities.idxmax()
    st.markdown("### ✅ Answer:")
    st.write(faq_data.loc[most_similar_idx, 'Answer'])
