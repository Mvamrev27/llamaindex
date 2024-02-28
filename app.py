import streamlit as st
import openai
from llama_index import LlamaIndex

# Load OpenAI key from secrets
openai.api_key = st.secrets["openai_key"]

llama = LlamaIndex()

st.title("Document Q&A Bot")

# Allow file uploads
uploaded_files = st.file_uploader("Upload documents", accept_multiple_files=True)

if uploaded_files:
    # Load files into LlamaIndex
    for file in uploaded_files:
        if file.name.endswith(".pdf"):
            llama.load_pdf(file) 
        elif file.name.endswith(".csv") or file.name.endswith(".xlsx"):
            llama.load_dataframe(read_csv_or_excel(file))

    st.success("Files loaded!")

# Input query  
query = st.text_input("Ask a question")
if query:
    # Get context from LlamaIndex
    tokens = llama.tokenize(query)
    vectors = llama.encode(tokens)
    hits = llama.query(vectors)
    context = " ".join([hit.text for hit in hits[:3]])

    # Generate answer with OpenAI
    prompt = f"Answer the question '{query}' based on this context: {context}"
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt, 
        max_tokens=100
    ).choices[0].text

    st.text(response)
