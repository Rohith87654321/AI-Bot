# Imports and downloads
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from urllib import request
import string
import os
import pickle
import re

'''
Web Crawler function to create a knowledge base using a custom web crawler.
Takes the starting url, the maximum links, and the maximum depth as parameters.
'''
def web_crawler(start_url, max_links, max_depth):

    #Keeping track of crawled urls and unique domains
    urls_visited = set()
    unique_domains = set()

    #Keeping track of relevant urls
    urls_relevant = []

    #Creating a queue of URLs to crawl
    queue = [(start_url, 0)]

    while queue and len(urls_relevant) < max_links:
        url, num = queue.pop(0)

        if url not in urls_visited and num <= max_depth:
            try:
                #Getting text from URL
                html = requests.get(url).text

                #Parsing the HTML content
                soup = BeautifulSoup(html, 'html.parser')

                #Storing the text in a file
                filename = f'{urlparse(url).hostname}.txt'
                with open(filename, 'w') as file:
                    for p in soup.find_all('p'):
                        file.write(p.get_text() + '\n')

                #Finding links in the page
                if num < max_depth:
                    for link in soup.find_all('a', href = True):
                      absolute_link = urljoin(url, link['href'])
                      domain = urlparse(absolute_link).hostname
                      if absolute_link not in urls_visited and domain not in unique_domains:
                        queue.append((absolute_link, num + 1))
                        unique_domains.add(domain)

                urls_relevant.append(url)
                urls_visited.add(url)
                print(f"Crawled: {url}")

            except requests.RequestException as e:
                print(f"Failed to get {url}")

    return urls_relevant[:max_links]

# Call the web_crawler function
start_url = 'https://www.biography.com/actors/leonardo-dicaprio'
urls_relevant = web_crawler(start_url, 25, 2)

'''
Cleaned files function to clean up the text files.
Takes the list of files as parameter
'''
def cleaned_files(file_list):
  cleaned_files = []

  for file in file_list:
    with open(file, 'r') as f:
      text = f.read()

    #Removing newlines from the text
    text = text.replace('\n', '')

    #Lowercase the text
    text = text.lower()

    #Tokenizing the text
    tokens = nltk.word_tokenize(text)

    #Removing stopwords and punctuation
    stop_words = set(nltk.corpus.stopwords.words('english'))
    new_tokens = [token for token in tokens if token.isalpha() and token not in stop_words]

    #Get the new cleaned text
    cleaned_text = '\n'.join(new_tokens)

    #Write sentences to a new file
    with open(f'cleaned_{file}', 'w') as output:
      output.write(cleaned_text)

    cleaned_files.append(cleaned_text)

  return cleaned_files

# Call the cleaned_files function
directory_files = [file for file in os.listdir() if file.endswith(".txt")]
cleaned_files_output = cleaned_files(directory_files)

'''
Important words function to get 40 important terms from the cleaned-up files.
Takes the list of cleaned files and the number of terms needed as a parameter.
'''
def important_words(cleaned_files, num_terms):

    #Geting the text from all files and combining it into one
    all_text = ' '.join(cleaned_files)

    #Tokenizeing the text
    tokens = nltk.word_tokenize(all_text)

    #Lowercase and remove stop words and punctuation
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = [token.lower() for token in tokens if token.isalpha() and token not in stop_words]

    #Lemmatizing the words
    lemmatizer = nltk.WordNetLemmatizer()
    terms = [lemmatizer.lemmatize(token) for token in tokens]

    #Create a tf-idf vectorizer
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([all_text])

    #Getting the feature names
    feature_names = tfidf_vectorizer.get_feature_names_out()

    # Get TF-IDF scores for each term
    tfidf_scores = tfidf_matrix.toarray()[0]

    #Creating terms dictionary and the tf-idf scores
    terms_dict = dict(zip(feature_names, tfidf_scores))

    #Sorting the terms based on descending order
    sorted_terms = sorted(terms_dict.items(), key=lambda x: x[1], reverse=True)

    #Getting the top important terms
    top_terms = [term[0] for term in sorted_terms[:num_terms]]

    return top_terms

# Calling the important_words function
important_terms = important_words(cleaned_files_output, 40)
print("Important terms determined by function: ")
print(important_terms)

# Function to get relevant sentences containing a the specific term from the text
def get_facts(text, term):
    # Tokenizing the text into sentences
    sentences = nltk.sent_tokenize(text)

    #Getting the sentences that contain the term
    relevant_sent = [sentence.strip() for sentence in sentences if term.lower() in sentence.lower()]

    return relevant_sent

#Function to update the knowledge base with facts from a provied URL
def update_knowledge_base(url, terms, knowledge_base):

    #Getting the html content from the url
    html = requests.get(url).text
    soup = BeautifulSoup(html, 'html.parser')

    #Get the content
    content = " ".join([paragraph.get_text() for paragraph in soup.find_all('p')])

    #Iterating through the given terms
    for i in terms:
      #Getting the relevant sentences for each term
      relevant_sent = get_facts(content, i)

      #Adding only one fact about the term to the knowledge base
      if relevant_sent:
        knowledge_base[i.lower()] = relevant_sent[0]

# Initializing the knowledge base dictionary
knowledge_base = {}

#Important terms that are relevant from the important_words() function
terms = ['dicaprio','film','one','movie','leonardo','million','time','actor','part','issue','diamond','issues','scorsese','hollywood','years','review','diamonds']

#Automarically adding the facts to the knowledge base
for url in urls_relevant:
    update_knowledge_base(url, terms, knowledge_base)

# Manually adding additional terms and facts to the knowledge base
knowledge_base['age'] = 'Leonardo DiCaprio is 49 years old.'
knowledge_base['height'] = 'Leonardo DiCaprio is approximately 6 feet tall.'
knowledge_base['producer'] = 'In 2013, Leonardo DiCaprio collaborated with  Martin Scorsese to star in and co-produce The Wolf of Wall Street.'
knowledge_base['oscar'] = 'Leonardo DiCaprio received the Oscar for Best Actor for the 2015 film, The Revenant.'
knowledge_base['awards'] = 'Leonardo DiCaprio\'s iconic film, Titanic, achieved immense success both critically and commercially. It received an impressive 14 Academy Award nominations and 11 wins.'
knowledge_base['titanic'] = 'Leonardo DiCaprio\'s Titanic was the first film to reach the billion dollar mark in international sales.'
knowledge_base['star'] = 'Leonardo DiCaprio has starred in  Quentin Tarantino\'s works such as Django Unchained and Once Upon a Time in Hollywood, in addition to other renowned films including Blood Diamond, Revolutionary Road, Inception, and The Great Gatsby.'
knowledge_base['perform'] = 'In preparation for his role in the 1993 film What’s Eating Gilbert Grape?, Leonardo DiCaprio spent several days studying the mannerisms of residents at homes for mentally ill teenagers and integrating them into his performance.'
knowledge_base['born'] = 'Leonardo DiCaprio was born on November 11, 1974, in Los Angeles, California, USA.'
knowledge_base['movies'] = 'Leonardo DiCaprio has appeared in several well-known films, such as Titanic, Inception, The Revenant, The Wolf of Wall Street, The Departed, Shutter Island, The Great Gatsby, and Once Upon a Time in Hollywood.'

#Saving the knowledge base as a pickle file
with open('knowledge_base.pkl', 'wb') as file:
    pickle.dump(knowledge_base, file)

#Printing the knowledge base
for term, fact in knowledge_base.items():
    print(f"{term.capitalize()}: {fact}")