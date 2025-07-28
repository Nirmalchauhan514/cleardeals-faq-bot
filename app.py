import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Load model and data
model = SentenceTransformer("all-MiniLM-L6-v2")
faq_data = pd.read_csv("faq_data.csv")
faq_data['Question'] = faq_data['Question'].astype(str)

# Create embeddings
question_embeddings = [model.encode(q) for q in faq_data['Question']]

# Page settings
st.set_page_config(page_title="Cleardeals FAQ Bot 🤖", layout="wide")

# Sidebar
with st.sidebar:
    st.image("https://media.giphy.com/media/3oriO0OEd9QIDdllqo/giphy.gif", caption="Ask Me Anything! 💬", use_column_width=True)
    st.title("📚 Cleardeals FAQs")
    st.markdown("👋 Hello! I'm here to help you solve any doubt you have about our services.")
    st.markdown("🔎 Type below or choose any question!")

# Search bar
st.markdown("## 🔍 Ask a Question")
typed_input = st.text_input("Type your query here...")

user_input = None
if typed_input:
    user_input = typed_input
    st.markdown(f"**You asked:** {typed_input}")

# Or select from listed buttons
st.markdown("---")
st.markdown("## 📋 Or Tap Any FAQ Below:")

for i, question in enumerate(faq_data['Question']):
    if st.button(f"❓ {question}", key=i):
        user_input = question

# Generate and show best-matched answer
if user_input:
    user_embedding = model.encode(user_input)
    scores = [util.pytorch_cos_sim(user_embedding, emb)[0][0].item() for emb in question_embeddings]
    best_match_index = scores.index(max(scores))
    answer = faq_data['Answer'][best_match_index]

    st.markdown("---")
    st.success(f"✅ **Answer:**\n\n{answer}")

# Footer
st.markdown("---")
st.markdown("<center><span style='color:gray;'>Built with ❤️ by Nirmal @ ClearDeals</span></center>", unsafe_allow_html=True)
