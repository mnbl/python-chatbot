import nltk
import random
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



# Opening a text file
f = open('chatbot.txt', 'r', errors= 'ignore')

# Reading opened text file
raw = f.read()

# converting file contents to lower case
raw = raw.lower()

# Downloading packages. First time only.
# nltk.download('punct')
# nltk.download('wordnet')

# Converting raw text into list of scentences (tokens)
sent_tokens = nltk.sent_tokenize(raw)

# Converting raw text to list of words (tokens)
word_token = nltk.word_tokenize(raw)

# print(sent_tokens)
# print(word_token)

# WordNet is a symentically-oriented dictionary of English included in NLTK
# Performing Lemmantization
lemmer = nltk.stem.WordNetLemmatizer()

#Lementizing tokens
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

# Removing punctuations from the read file
remove_punctuation = dict((ord(punct), None) for punct in string.punctuation)

# Returning normalized tokens
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punctuation)))

# Determining user
def response(user_input):
    bot_response = 'Bot: '
    sent_tokens.append(user_input)

    tfIdfVec = TfidfVectorizer(stop_words="english", tokenizer=LemNormalize)
    vect = tfIdfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(vect[-1], vect)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    result = flat[-2]

    if (result == 0):
        bot_response = "Sorry! Didn't get that"
        return bot_response
    else:
        bot_response = bot_response + sent_tokens[idx]
        return bot_response
    
flag = True
print("Hi! I am a chatbot. Enter your question about Apple or type bye to exit")

while(flag == True):
    user_input = input("User: ")
    user_input = user_input.lower()
    if (user_input != 'bye'):
        if (user_input == 'thanks' or user_input == 'thank you'):
            flag = False
            print("Bot: You are welcome")
        else:
            print(response(user_input))
            sent_tokens.remove(user_input)
    else:
        flag = False
        print("Bot: Exiting Now")