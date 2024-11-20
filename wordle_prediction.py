import streamlit as st
import pandas as pd
import numpy as np
import pickle
import magic
import os
from streamlit_autorefresh import st_autorefresh

st.set_page_config(
    page_title="Cheatdle",
    page_icon="🟩"
)

main, forest, analysis = st.tabs(["main", "forest", "analysis"])

with main:
    st.header("Header")

    def get_stats():
        return {
            'Top Picks': ['tones', 'dotes', 'potes', 'cotes', 'topes', 'poets', 'terms', 'stone', 'coste', 'certs', 'temp', 'temp'],
            'E[Info.]': [4.86, 4.85, 4.84, 4.83, 4.82, 4.81, 4.81, 4.79, 4.78, 4.78, 4.78, 4.78],
            'p(word)': [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
        }  # placeholder

    [wordle, stats] = st.columns([0.6, 0.4])
    with stats:
        st.subheader('Word Suggestions')

        if st.checkbox(label="Show Suggestions"):
            stats = get_stats()

            df = pd.DataFrame(stats)

            st.dataframe(df, width=300, height=458, hide_index=True)

    with wordle:
        st.subheader('Wordle Results')
        with st.container(border=True, height=530):
            count = st_autorefresh(interval=2000, limit=100, key="imagereloader")
            path = 'captures/capture.png'

            if os.path.isfile(path) and magic.from_file(path) != 'empty':
                path = 'captures/capture.png'
            elif count:
                path = 'captures/blank.png'

            st.image(path)


with forest:
    # Create a page header
    st.header("Wordle Score Prediction")

    with st.form("my_form"):
        word = st.text_input(
            "Enter any five-letter combination",
            max_chars=5
        )
        occurrence = st.number_input(
            "Enter the word's frequency:", max_value=0.00)

        submitted = st.form_submit_button("Predict")

    if submitted:
        tweets = pd.read_csv("data/tweets.csv")
        words = pd.read_csv("data/words_freq.csv")

        tweets["score"] = tweets["tweet_text"].str[11]
        tweets["score"] = pd.to_numeric(tweets['score'], errors='coerce')
        tweets.rename(columns={"wordle_id": "day"}, inplace=True)

        words.dropna(inplace=True)
        words["day"] = pd.to_numeric(words['day'], errors='coerce')

        df = pd.merge(words, tweets, on='day')
        df.drop(columns=['tweet_id'], inplace=True)

        filename = 'wordle_prediction.pkl'
        model = pickle.load(open(filename, 'rb'))

        # For any given word:

        #    1. Put the word in lower case
        #    2. Extract each letter in the word and make it it's own column
        #    3. Convert to ASCII number using ord() function
        #    4. subtract 96 to simplify char to number representation (a = 1, b = 2, c = 3, ...)

        def predict_score(word, occurrence):

            if (not word.isalpha() or len(word) != 5):
                raise Exception(
                    "Invalid word format. Please enter a five letter word using only alphabetic characters.")

            df = pd.DataFrame()
            df["word"] = [word]
            df["letter_1"] = df["word"].str.lower().str[0].apply(ord) - 96
            df["letter_2"] = df["word"].str.lower().str[1].apply(ord) - 96
            df["letter_3"] = df["word"].str.lower().str[2].apply(ord) - 96
            df["letter_4"] = df["word"].str.lower().str[3].apply(ord) - 96
            df["letter_5"] = df["word"].str.lower().str[4].apply(ord) - 96

            df["occurrence"] = [occurrence]
            df.drop(columns=["word"], inplace=True)

            return model.predict(df)

        averages = df.groupby("word", as_index=False)['score'].mean()

        def get_scores(word, base=None, power=None):

            prediction = predict_score(word, occurrence)[0]

            # If word isn't found in tweet data, None is returned for the average score
            average = None
            if word in averages["word"].values:
                average = averages[averages["word"] == word]["score"][0]

            return prediction, average

        prediction, average = get_scores(word, occurrence)

        st.write(f"Word: {word}")
        st.write(
            "Predicted average score via random forests: \t{:0.2f}".format(prediction))
        # Print average score according to tweet data if the word exists in it
        st.write(("No data found for this word in tweet data." if average ==
                 None else "Average score via tweet data: \t\t\t{:0.2f}".format(average)))

with analysis:
    st.header("Sentiment Analysis")
