# Imports and downloads
import random
import pickle
import requests
from bs4 import BeautifulSoup
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the knowledge base from the pickle file
with open('knowledge_base.pkl', 'rb') as file:
    knowledge_base = pickle.load(file)

# Initializing a dictionary to store the user models
user_models = {}

# Function to generate responses based on user input
def chatbot(user_input):

    #Checking if the user says hello
    if 'hello' in user_input.lower():
      return random.choice(['Hi!', 'Hello!'])

    # Checking if the user has interacted with the chatbot before
    if user_input not in user_models:
        user_models[user_input] = {'name': '', 'likes': set(), 'dislikes': set()}

    user_model = user_models[user_input]

    # Checking if the user input is their name
    if 'name is' in user_input.lower():
        user_model['name'] = user_input.split('name is')[-1].strip()

    #Parsing the user input with tokenization
    tokens = nltk.word_tokenize(user_input.lower())

    # Updating the user model with the user's likes and dislikes based on their input
    for token in tokens:
      if token == 'like':
        user_model['likes'].add(user_input)
      elif token == 'dislike':
        user_model['dislikes'].add(user_input)

    # Checking if the user wants to exit
    if user_input.lower() == 'exit':
      if user_model['name']:
        return f"Goodbye {user_model['name']}! Have a great day!"
      else:
        return "Goodbye! Have a great day!"

    # Checking if the user input matches with knowledge base terms
    response = ""
    full_user_input = ' '.join(tokens)
    for word in knowledge_base:
      if word in full_user_input:
        response = knowledge_base[word]
        break

    # If a matching term is found, return the knowledge base response, if not then generate a random response
    if response:
        return response
    else:
        responses = [
            "Sorry, I couldn't find the information you're looking for. Please rephrase or ask something else.",
            "I'm not sure I understand. Could you provide more information?",
            "I'm still learning! How about asking me something simpler?"
        ]
        return random.choice(responses)

    return response

#Running the chatbot
print("Chatbot: Hello, I'm a chatbot about Leonardo DiCaprio. You can exit the chatbot at any time by entering 'exit'. What is your name?")
user_input = input("User: ")

if 'name is' in user_input.lower():
    #Getting the name of the user if the input contains 'name is'
    user_name = user_input.split('name is')[-1].strip()
else:
    # Using the user input directly as the name
    user_name = user_input

print(f"Chatbot: Hello {user_name}, What can I help you with today?")

while True:
    user_input = input("User: ")
    response = chatbot(user_input)
    print("Chatbot:", response)