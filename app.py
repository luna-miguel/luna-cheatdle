import os
import json
import magic
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from textblob import TextBlob

st.set_page_config(
    page_title="Cheatdle",
    page_icon="🟩"
)

main, forest, analysis = st.tabs(["main", "forest", "analysis"])

with main:
    st.header("Sentiment Analysis")
    st.markdown(
    """
    Enter any **5-letter Wordle word**, and we'll show how people on Twitter felt about it! 🎉
    """)

    # Load datasets
    try:
        words_freq = pd.read_csv("data/words_freq.csv")
        tweets = pd.read_csv("data/tweeets.csv")
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
        
    