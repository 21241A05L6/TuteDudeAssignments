import streamlit as st
from transformers import pipeline

# ---------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------
st.set_page_config(
    page_title="GenAI Assistant",
    page_icon="🤖",
    layout="centered"
)

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------
st.title("🤖 GenAI Assistant")

st.write(
    "This application uses a real AI language model from Hugging Face "
    "to answer questions and generate text."
)

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------
@st.cache_resource
def load_model():

    generator = pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )

    return generator


model = load_model()

# ---------------------------------------------------
# SELECT TASK
# ---------------------------------------------------
task = st.selectbox(
    "Choose AI Task",
    [
        "Question Answering",
        "Summarization",
        "Explain Like a Beginner"
    ]
)

# ---------------------------------------------------
# USER INPUT
# ---------------------------------------------------
user_input = st.text_area(
    "Enter your text:",
    height=200,
    placeholder="Type something here..."
)

# ---------------------------------------------------
# GENERATE RESPONSE
# ---------------------------------------------------
if st.button("Generate AI Response"):

    # Input validation
    if user_input.strip() == "":
        st.warning("Please enter some text.")

    else:

        try:

            with st.spinner("Generating response..."):

                # ---------------------------------------------------
                # PROMPT ENGINEERING
                # ---------------------------------------------------

                if task == "Question Answering":

                    prompt = f"""
                    Answer the following question clearly:

                    Question:
                    {user_input}

                    Answer:
                    """

                elif task == "Summarization":

                    prompt = f"""
                    Summarize the following text in short:

                    {user_input}

                    Summary:
                    """

                elif task == "Explain Like a Beginner":

                    prompt = f"""
                    Explain this topic like teaching a small child:

                    {user_input}

                    Explanation:
                    """

                # ---------------------------------------------------
                # GENERATE AI OUTPUT
                # ---------------------------------------------------

                result = model(
                    prompt,
                    max_new_tokens=80,
                    temperature=0.7,
                    do_sample=True,
                    truncation=True
                )

                # Extract generated text
                response = result[0]["generated_text"]

                # Remove prompt from response
                response = response.replace(prompt, "").strip()

                # ---------------------------------------------------
                # DISPLAY OUTPUT
                # ---------------------------------------------------

                st.success("AI Response Generated Successfully!")

                st.write("## Generated Response")

                st.write(response)

        except Exception as e:

            st.error(f"Error: {e}")

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
st.sidebar.title("📌 About Project")

st.sidebar.info(
    """
    This is a Generative AI application developed using:

    - Streamlit
    - Hugging Face Transformers
    - TinyLlama LLM

    Features:
    - Question Answering
    - Summarization
    - Beginner Explanations
    """
)

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("---")

st.caption("Assignment 39 - GenAI App Deployment")