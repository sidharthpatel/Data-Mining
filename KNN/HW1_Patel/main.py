import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to remove all html tags from a given string
def remove_html(string):
    return re.sub('<.*?>', '', string)

# Function to remove all punctuations from the given string
def remove_punctuations(string):
    output = ''
    for i in string:
        if i.isalnum():
            output = output + i
        else:
            output = output + ' '
    return output

# Function to get rid of any extra spaces after pre-processing the removal of html tags and punctuations
def remove_extra_spaces(string):
    return re.sub(" +", " ", string)

# A generic preprocess function that calls all of the above functions to assist in preprocessing
def preprocess(text_file):
    for i in range(len(text_file)):
        # Lowercase the reviews
        text_file[i] = text_file[i].lower()
        # Remove Html tags
        text_file[i] = remove_html(text_file[i])
        # Remove Punctuations
        text_file[i] = remove_punctuations(text_file[i])
        # Remove Extra Spaces
        text_file[i] = remove_extra_spaces(text_file[i])
        # Convert the strings to an array
        tokenizers = text_file[i].split()
        text_file[i] = tokenizers
        # Remove Stopwords
        stop = set(stopwords.words("english"))
        text_file[i] = [eachWord for eachWord in text_file[i] if eachWord not in stop]
        # Lemmatization
        lemma = WordNetLemmatizer()
        for j in range(len(text_file[i])):
            text_file[i][j] = lemma.lemmatize(text_file[i][j])
    return text_file

# ------------------ Start -----------------

# Reading train data into a DataFrame
train_file = pd.read_csv("train_data.txt", delimiter="\t")
# Opening the test data file
test_file = open("test_data.txt", 'r', encoding='utf-8')

# Initialized an array to store all the test data
test_data = []

# Traversing the test_data.txt file and appending each line in the above array
with test_file as td:
    for data in td:
        test_data.append(data)

# To make processing and understanding data easier, I am converting all the data into a list/ array

# Separating the train data into a list of reviews
train_review = list(np.array(train_file.review))
# Separating the train data into a list of sentiments
train_sentiment = list(np.array(train_file.sentiment))

# Pre-processing the Train and Test data by calling the above function
cleaned_train_review = preprocess(train_review)
cleaned_test_review = preprocess(test_data)

# Converting list of lists generated during the pre-process back to list of strings
for i in range(len(cleaned_train_review)):
    string = " ".join([str(word) for word in cleaned_train_review[i]])
    cleaned_train_review[i] = string

for i in range(len(cleaned_test_review)):
    string = " ".join([str(word) for word in cleaned_test_review[i]])
    cleaned_test_review[i] = string

# Converted test array to a Pandas DataFrame to fix issue of passing arguments in the cosine similarity function.
test_dataframe = pd.DataFrame(cleaned_test_review)

# Vectorised the data converting words into weighted numbers using TF-IDF Vectorizer.
vectorizer = TfidfVectorizer(analyzer='word')
vectored_train_review = vectorizer.fit_transform(cleaned_train_review)
train_frame = pd.DataFrame(data=vectored_train_review.toarray(), columns=vectorizer.get_feature_names(), index=train_file.index)
vectored_test_review = vectorizer.transform(cleaned_test_review)
test_frame = pd.DataFrame(data=vectored_test_review.toarray(), index=test_dataframe.index, columns=vectorizer.get_feature_names())

# Generated a 15000 x 14999 matrix
matrix = 1 - cosine_similarity(test_frame, train_frame)

# Used k = 9 to predict the nearest neighbors and decide what the test sentiments would be.
k = 9
result = []
for train in matrix:
    output = []
    temp = list(train)
    for i in range(k):
        search = np.array(temp)
        min_val = np.min(search[np.nonzero(search)])
        index = temp.index(min_val)
        sentiment = train_sentiment[index]
        output.append(sentiment)
        temp[index] = 0
    score = sum(output)
    if score > 0:
        result.append(1)
    else:
        result.append(-1)

# Writes all the scores from the KNN Algorithm to a format.txt file.
with open('format.txt', 'w', encoding="utf-8") as finalFile:
    finalFile.writelines("%s\n" % place for place in result)