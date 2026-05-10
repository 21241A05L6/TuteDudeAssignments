# Assignment 39 - GenAI App Deployment

# GenAI Assistant using TinyLlama

## Project Overview

In this assignment, I developed and deployed a real Generative AI application using Streamlit and Hugging Face Transformers.

The application uses the TinyLlama language model to perform different AI-based tasks such as:

- Question Answering
- Text Summarization
- Explaining concepts in simple language

The main goal of this assignment was to understand how GenAI applications are deployed on cloud platforms like Streamlit Cloud and Hugging Face Spaces.

---

# Why I Chose TinyLlama

I initially explored larger AI models and OpenAI APIs, but those required paid API keys or high system resources.

Since this project was developed inside GitHub Codespaces and free deployment platforms, I selected TinyLlama because:

- It is lightweight
- Free to use
- Works properly on limited resources
- Easy to deploy on Streamlit Cloud and Hugging Face Spaces
- Still demonstrates real LLM integration and prompt engineering concepts

The purpose of using this smaller AI model was to clearly demonstrate successful deployment and integration of a real Generative AI model within the available system limitations.

---

# Technologies Used

- Python
- Streamlit
- Hugging Face Transformers
- TinyLlama LLM
- GitHub Codespaces

---

# Features Implemented

## 1. Question Answering

The user can ask any question and the AI model generates a response.

Example:

Question:
What is Artificial Intelligence?

AI Response:
Artificial Intelligence is a field of computer science that enables machines to simulate human intelligence.

---

## 2. Text Summarization

The application can summarize long text into shorter content.

---

## 3. Beginner-Friendly Explanation

The model can explain difficult topics in simple words like teaching a child.

---

## 4. Prompt Engineering

Different prompts were created for different tasks.

Examples:

- Answer clearly
- Summarize briefly
- Explain like a beginner

This helped me understand how prompts affect AI-generated responses.

---

## 5. Error Handling

I added:

- Empty input validation
- Exception handling using try-except blocks
- User-friendly warning and error messages

This improved the reliability of the application.

---

# Project Structure

```text
ASSIGNMENT_39_GENAI/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
