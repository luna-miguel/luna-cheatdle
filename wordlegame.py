import pygame
import random
import json
import streamlit as st
from PIL import Image

# Function to load dictionary
def load_dict(file_name):
    with open(file_name, 'r') as f:
        words = [line.strip() for line in f.readlines()]
        return [word.upper() for word in words]

# Initialize game variables
DICT_GUESSING = load_dict('data/wordle-answers.txt')
DICT_ANSWERS = load_dict('data/valid-wordle-words.txt')
ANSWER = random.choice(DICT_ANSWERS)

WIDTH = 600
HEIGHT = 700
MARGIN = 10
T_MARGIN = 100
B_MARGIN = 100
LR_MARGIN = 100

GREY = (70, 70, 80)
GREEN = (6, 214, 160)
YELLOW = (255, 209, 102)

pygame.init()
pygame.font.init()
SQ_SIZE = (WIDTH - 4 * MARGIN - 2 * LR_MARGIN) // 5
FONT = pygame.font.SysFont("free sans bold", SQ_SIZE)
FONT_SMALL = pygame.font.SysFont("free sans bold", SQ_SIZE // 2)

# Streamlit state initialization
if "guesses" not in st.session_state:
    st.session_state["guesses"] = []
    st.session_state["input"] = ""
    st.session_state["answer"] = ANSWER
    st.session_state["unguessed"] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    st.session_state["game_over"] = False

# Function to determine unguessed letters
def determine_unguessed_letters(guesses):
    guessed_letters = "".join(guesses)
    return "".join([letter for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if letter not in guessed_letters])

# Function to determine color of letters
def determine_color(guess, j):
    letter = guess[j]
    if letter == st.session_state["answer"][j]:
        return GREEN
    elif letter in st.session_state["answer"]:
        return YELLOW
    else:
        return GREY

# Function to draw guesses
def draw_guesses(surface):
    y = T_MARGIN
    for i in range(6):
        x = LR_MARGIN
        for j in range(5):
            square = pygame.Rect(x, y, SQ_SIZE, SQ_SIZE)
            pygame.draw.rect(surface, GREY, square, width=2, border_radius=3)

            if i < len(st.session_state["guesses"]):
                color = determine_color(st.session_state["guesses"][i], j)
                pygame.draw.rect(surface, color, square, border_radius=3)
                letter = FONT.render(st.session_state["guesses"][i][j], False, (255, 255, 255))
                surface.blit(letter, letter.get_rect(center=(x + SQ_SIZE // 2, y + SQ_SIZE // 2)))

            if i == len(st.session_state["guesses"]) and j < len(st.session_state["input"]):
                letter = FONT.render(st.session_state["input"][j], False, GREY)
                surface.blit(letter, letter.get_rect(center=(x + SQ_SIZE // 2, y + SQ_SIZE // 2)))

            x += SQ_SIZE + MARGIN
        y += SQ_SIZE + MARGIN

# Function to render the game frame
def render_frame():
    surface = pygame.Surface((WIDTH, HEIGHT))
    surface.fill("white")
    letters = FONT_SMALL.render(st.session_state["unguessed"], False, GREY)
    surface.blit(letters, letters.get_rect(center=(WIDTH // 2, T_MARGIN // 2)))
    draw_guesses(surface)
    return pygame.surfarray.array3d(surface).swapaxes(0, 1)

# Streamlit interface
st.title("Wordle in Streamlit")

if st.session_state["game_over"]:
    st.success("Game Over!")
    st.write(f"The correct word was: {st.session_state['answer']}")

frame = render_frame()
frame_image = Image.fromarray(frame)
st.image(frame_image)

# Input field for guesses
guess = st.text_input("Enter your guess:", max_chars=5).upper()
if st.button("Submit Guess") and len(guess) == 5:
    if guess in DICT_GUESSING:
        st.session_state["guesses"].append(guess)
        st.session_state["unguessed"] = determine_unguessed_letters(st.session_state["guesses"])
        st.session_state["game_over"] = (guess == st.session_state["answer"] or len(st.session_state["guesses"]) == 6)

if st.button("Restart Game"):
    st.session_state["guesses"] = []
    st.session_state["input"] = ""
    st.session_state["answer"] = random.choice(DICT_ANSWERS)
    st.session_state["unguessed"] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    st.session_state["game_over"] = False