import streamlit as st

st.set_page_config(page_title="GenAI App", page_icon="🤖")

st.title("🤖 GenAI Demo App")
st.write("This is a simple deployed GenAI app using Streamlit Cloud.")

user_input = st.text_input("Ask something:")

if user_input:
    st.write("You asked:", user_input)
    st.write("Response:", "This is a placeholder GenAI response.")