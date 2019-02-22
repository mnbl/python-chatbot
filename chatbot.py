import nltk
import random
import string


# Opening a text file
f = open('chatbot.txt', 'r', errors= 'ignore')

# Reading opened text file
raw = f.read()

# converting file contents to lower case
raw = raw.lower()

# Downloading packages. First time only.
# nltk.download('punct')
# nltk.download('wordnet')

# Converting raw text into list of scentences
sent_tokens = nltk.sent_tokenize(raw)

# Converting raw text to list of words
word_token = nltk.word_tokenize(raw)

print(sent_tokens)
print(word_token)