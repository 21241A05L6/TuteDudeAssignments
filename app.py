import streamlit as st

st.set_page_config(page_title="GenAI Deployment App")

st.title("🧠 GenAI Deployment Assignment")

user_input = st.text_input("Enter your name")

if st.button("Generate Greeting"):
    st.success(f"Hello, {user_input}! Your GenAI app is deployed successfully.")