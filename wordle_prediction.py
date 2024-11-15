import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Cheatdle",
    page_icon="🟩",
)

# Create a page header
st.header("Wordle Score Prediction")

word = st.text_input(
        "Enter any five-letter combination",
        max_chars=5
    )

base = None
power = None

freq = st.checkbox('Frequency of word known?')
if freq:
    a, b = st.columns(2)
    with a:
        base = st.number_input("Coefficient:", min_value = 0.00, max_value = 9.99)
    with b:
        power = st.number_input("Power of 10:", value = -3, max_value = -3)

submitButton = st.button('Calculate', type="primary")

if submitButton:
    tweets = pd.read_csv("tweets.csv")
    words = pd.read_csv("words_freq.csv")

    tweets["score"] = tweets["tweet_text"].str[11]
    tweets["score"] = pd.to_numeric(tweets['score'], errors='coerce')
    tweets.rename(columns={"wordle_id": "day"}, inplace=True)

    words.dropna(inplace=True)
    words["day"] = pd.to_numeric(words['day'], errors='coerce')

    df = pd.merge(words, tweets, on='day')
    df.drop(columns=['tweet_id'], inplace=True)

    # For any given word:

    #    1. Put the word in lower case
    #    2. Extract each letter in the word and make it it's own column
    #    3. Convert to ASCII number using ord() function
    #    4. subtract 96 to simplify char to number representation (a = 1, b = 2, c = 3, ...)

    def predict_score (word, base=None, power=None):
        
        df = pd.DataFrame()
        df["word"] = [ word ]
        df["const"] = 1.0
        
        df["letter_1"] = df["word"].str.lower().str[0].apply(ord) - 96
        df["letter_2"] = df["word"].str.lower().str[1].apply(ord) - 96
        df["letter_3"] = df["word"].str.lower().str[2].apply(ord) - 96
        df["letter_4"] = df["word"].str.lower().str[3].apply(ord) - 96
        df["letter_5"] = df["word"].str.lower().str[4].apply(ord) - 96
        
        df.drop(columns=["word"], inplace=True)
        # Calculate weight of word frequency towards score if it is known
        if (base != None and power != None): 
            occurrence_weight = (16.7055 * (base * (10 ** power)))  
        else: 
            occurrence_weight = 0

        return 4.0984   + (-0.0002 * df["letter_1"]) \
                        + (0.0016 * df["letter_2"]) \
                        + (-0.0023 * df["letter_3"]) \
                        + (0.0023 * df["letter_4"]) \
                        + (0.0009 * df["letter_5"]) + occurrence_weight


    averages = df.groupby("word", as_index=False)['score'].mean()

    def get_scores(word, base=None, power=None):
        
        prediction = predict_score (word, base, power) [0]

        # If word isn't found in tweet data, None is returned for the average score
        average = None
        if word in averages["word"].values:
            average = averages[averages["word"] == word]["score"][0]

        return prediction, average
        
    prediction, average = get_scores(word, base, power)

    st.write(f"Word: {word}\n")
    st.write("Predicted score via linear regression: \t{:0.2f}".format(prediction))
    # Print average score according to tweet data if the word exists in it
    st.write(("No data found for this word in tweet data." if average == None else "Average score via tweet data: \t\t{:0.2f}".format(average)))





