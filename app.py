import os
import json
import magic
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from textblob import TextBlob
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings  # Updated import
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

st.set_page_config(
    page_title="Cheatdle",
    page_icon="🟩"
)

sentiment, rag = st.tabs(["sentiment", "rag"])

with sentiment:
    st.header("Sentiment Analysis")
    st.markdown(
    """
    Enter any **5-letter Wordle word**, and we'll show how people on Twitter felt about it! 🎉
    """)

    # Load datasets
    try:
        words_freq = pd.read_csv("data/words_freq.csv")
        tweets = pd.read_csv("data/tweets.csv")
    except FileNotFoundError as e:
        st.error(f"Error: {e}. Ensure the file paths are correct.")
        st.stop()

    # Input Word
    word = st.text_input("Enter a 5-letter Wordle word:", max_chars=5).lower()

    if word:
        # Validate the word
        if not word.isalpha() or len(word) != 5:
            st.error("Please enter a valid 5-letter word.")
        else:
            # Check if word exists in dataset
            word_entry = words_freq[words_freq["word"].str.lower() == word]

            if word_entry.empty:
                st.error(f"The word '{word}' was not found in the dataset.")
            else:
                # Get Wordle day and filter tweets
                wordle_day = int(word_entry.iloc[0]["day"])
                wordle_tweets = tweets[tweets["wordle_id"] == wordle_day]

                if wordle_tweets.empty:
                    st.error(f"No tweets found for Wordle #{wordle_day}.")
                else:
                    st.success(f"Analyzing tweets for Wordle #{wordle_day}...")

                    # Sentiment Analysis
                    sentiments = {"positive": 0, "neutral": 0, "negative": 0}
                    polarity_scores = []

                    for _, row in wordle_tweets.iterrows():
                        text = row["tweet_text"]
                        # Skip grid-only tweets
                        if text.count('\n') <= 1 and text.startswith("Wordle"):
                            continue

                        cleaned_text = ' '.join([
                            line for line in text.split('\n')
                            if not line.strip().startswith(('Wordle', '⬛', '⬜', '🟨', '🟩'))
                        ])

                        if cleaned_text.strip():
                            analysis = TextBlob(cleaned_text)
                            polarity = analysis.sentiment.polarity
                            polarity_scores.append(polarity)

                            if polarity > 0:
                                sentiments["positive"] += 1
                            elif polarity < 0:
                                sentiments["negative"] += 1
                            else:
                                sentiments["neutral"] += 1

                    total = sum(sentiments.values())

                    # Results Display
                    if total == 0:
                        st.warning("No valid tweets found for analysis.")
                    else:
                        avg_sentiment = sum(polarity_scores) / len(polarity_scores)
                        sentiment_label = "😊 Positive" if avg_sentiment > 0 else "😐 Neutral" if avg_sentiment == 0 else "😟 Negative"

                        st.subheader(f"Results for '{word}' (Wordle #{wordle_day}):")
                        st.markdown(f"**Total Tweets Analyzed:** {total}")
                        st.markdown(f"**Average Sentiment:** {sentiment_label} ({avg_sentiment:.3f})")
                        st.markdown("### Sentiment Breakdown:")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Positive 😊", sentiments["positive"])
                        col2.metric("Neutral 😐", sentiments["neutral"])
                        col3.metric("Negative 😟", sentiments["negative"])

                        # Sentiment Slider
                        st.markdown("### Sentiment Visualization:")
                        st.slider(
                            label="Average Sentiment Score",
                            min_value=-1.0,
                            max_value=1.0,
                            value=float(avg_sentiment),
                            step=0.01,
                            disabled=True,
                        )

                        # Additional Stats
                        st.markdown("### Additional Insights:")
                        st.markdown(f"- **Positive Percentage:** {sentiments['positive'] / total * 100:.2f}%")
                        st.markdown(f"- **Neutral Percentage:** {sentiments['neutral'] / total * 100:.2f}%")
                        st.markdown(f"- **Negative Percentage:** {sentiments['negative'] / total * 100:.2f}%")
        
with rag:
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    # Set environment variable to handle tokenizer warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Check for API key before proceeding
    if not api_key:
        st.error("OpenAI API key not found! Please set OPENAI_API_KEY in your .env file")
        st.write("1. Create a .env file in your project directory")
        st.write("2. Add your OpenAI API key like this: OPENAI_API_KEY=sk-your_api_key_here")
        st.write("3. Make sure the .env file is in the same directory as your Python script")
        st.stop()

    @st.cache_resource
    def initialize_qa_chain():
        try:
            # Get the absolute path to the PDF relative to the script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            pdfpath = os.path.join(script_dir, "data/CTP Project Design Doc (3).pdf")
            
            # Check if PDF exists
            if not os.path.exists(pdfpath):
                st.error(f"PDF file not found at: {pdfpath}")
                st.stop()
                
            st.write(f"Loading PDF from: {pdfpath}")
            
            # Load PDF
            loader = PyPDFLoader(pdfpath)
            pages = loader.load()
            
            # Create embeddings
            embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L12-v2",
                model_kwargs={'device': 'cpu'}
            )
            
            # Create vector store
            vectorstore = FAISS.from_documents(pages, embeddings)
            
            # Initialize ChatOpenAI with explicit API key
            llm = ChatOpenAI(
                temperature=0.7,
                api_key=api_key,
                model="gpt-3.5-turbo",
                max_tokens=100,
            )
            
            # Create the QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(),
            )
            
            return qa_chain
            
        except Exception as e:
            st.error(f"Error in initialize_qa_chain: {str(e)}")
            raise e

    # Page title
    st.title("Ask about our Wordle Final Project")

    # Initialize session state for messages
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Initialize QA chain
    try:
        with st.spinner("Loading PDF and initializing QA system..."):
            qa_chain = initialize_qa_chain()
        st.success("QA system initialized successfully!")
    except Exception as e:
        st.error(f"Error initializing QA chain: {str(e)}")
        st.stop()

    # Placeholder for chat messages
    chat_placeholder = st.empty()

    # Render chat history dynamically
    with chat_placeholder.container():
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Chat input with debouncing
    if "last_prompt" not in st.session_state:
        st.session_state.last_prompt = ""

    prompt = st.chat_input("Ask a question about the Wordle Final Project")

    if prompt and prompt != st.session_state.last_prompt:
        st.session_state.last_prompt = prompt

        # Display user message immediately
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Refresh chat history
        with chat_placeholder.container():
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # Process the input and get a response
        try:
            with st.spinner("Searching document for answer..."):
                response = qa_chain.invoke(prompt)

            result = response["result"]

            # Display assistant response
            st.session_state.messages.append({"role": "assistant", "content": result})

            # Refresh chat history again
            with chat_placeholder.container():
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

        except Exception as e:
            st.error(f"Error processing question: {str(e)}")