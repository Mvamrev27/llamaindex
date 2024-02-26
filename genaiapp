# streamlit_app.py
import streamlit as st
import pandas as pd
import openai
import tempfile

# Read OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["openai_secret_key"]

def analyze_text_with_openai(text):
    # Function to call OpenAI's API for Sentiment Analysis and Topic Classification
    try:
        response = openai.Completion.create(
            model="gpt-3.5-turbo",  # You might need to update the model name based on availability
            prompt=f"Analyze the sentiment and categorize the topics of the following text: {text}",
            temperature=0.5,
            max_tokens=60,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def process_csv(file):
    df = pd.read_csv(file)
    # Assuming 'text_column' is the column with text data
    df['analysis'] = df['text_column'].apply(analyze_text_with_openai)
    return df

# Streamlit UI setup
st.title('LlamaIndexApp: Sentiment Analysis and Topic Classification with OpenAI')

uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    with st.spinner('Processing...'):
        output_df = process_csv(uploaded_file)
        output_file_path = tempfile.NamedTemporaryFile(delete=False, suffix='.csv').name
        output_df.to_csv(output_file_path, index=False)
        st.success('Processing complete. Download the output CSV below.')
        st.download_button(label="Download Output CSV", data=output_df.to_csv(index=False), file_name="processed_output.csv", mime='text/csv')

    # Simplified RAG system based on OpenAI's response
    st.header("Chat with LlamaIndexApp")
    user_input = st.text_input("Ask me anything based on the processed data:")
    if user_input:
        response = analyze_text_with_openai(user_input)
        st.text_area("Response:", value=response, height=100, max_chars=None, key=None)
