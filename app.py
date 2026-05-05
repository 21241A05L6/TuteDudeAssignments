import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="AI Text Summarizer")

st.title("🧠 GenAI Text Summarizer")

st.write("Enter a paragraph and generate AI summary using Transformers.")

@st.cache_resource
def load_model():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_model()

user_input = st.text_area("Enter Text")

if st.button("Generate Summary"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        try:
            summary = summarizer(
                user_input,
                max_length=80,
                min_length=20,
                do_sample=False
            )

            st.success("Summary Generated Successfully!")

            st.write(summary[0]['summary_text'])

        except Exception as e:
            st.error(f"Error: {e}")