import string
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.wordnet import WordNetLemmatizer

def remove_html(string):
    return re.sub('<.*?>', '', string)

def remove_punctuations(string):
    rem = ''
    for i in string:
        if i.isalnum():
            rem = rem + i
        else:
            rem = rem + ' '
    return rem

def remove_extra_spaces(string):
    return re.sub(" +", " ", string)

def remove_stopwords(string):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(string)
    return [w for w in words if w not in stop_words]

def stemmer(string):
    stemmer = SnowballStemmer(language="english")
    # return " ".join([stemmer.stem(w) for w in string])
    stems = []
    for w in string:
        x = stemmer.stem(w)
        stems.append(x)
    return stems

def preprocess_train_data(text_file):
    for element in range(len(text_file)):
        text_file[element] = text_file[element].lower()
        text_file[element] = remove_html(text_file[element])
        text_file[element] = remove_punctuations(text_file[element])
        text_file[element] = remove_extra_spaces(text_file[element])
        tokenizers = text_file[element].split()
        text_file[element] = tokenizers
        stop = set(stopwords.words("english"))
        text_file[element] = [eachWord for eachWord in text_file[element] if eachWord not in stop]
        lemma = WordNetLemmatizer()
        for subEle in range(len(text_file[element])):
            text_file[element][subEle] = lemma.lemmatize(text_file[element][subEle])
    return text_file

def knn(listType, word):
    score = 0
    for sublist in listType:
        # if score < 5:
        count = sublist.count(word)
        score = score + count
        # else:
            # break
    return score

# ------------------
print("Reading Train and Test files...")
train_file = pd.read_csv("train_data.txt", delimiter="\t")
test_file = open("test_data.txt", 'r', encoding='utf-8')

print("Appending Testing data to Array...")
test_data = []
with test_file as td:
    for data in td:
        test_data.append(data)

pos_rev = []
neg_rev = []
sentiments = list(np.array(train_file.sentiment))
reviews = list(np.array(train_file.review))

print("Separating positive and negative reviews...")
for ele in range(len(sentiments)):
    if sentiments[ele] == 1:
        pos_rev.append(reviews[ele])
    else:
        neg_rev.append(reviews[ele])

print("Pre-Processing postive and negative reviews...")
processed_pos_rev = preprocess_train_data(pos_rev)
processed_neg_rev = preprocess_train_data(neg_rev)

print("Pre-Processing test data reviews...")
processed_test_rev = preprocess_train_data(test_data)

print("Running KNN Model...")
score = []
for line in processed_test_rev:
    positive_score = 0
    negative_score = 0
    for word in line:
        negative_score = negative_score + knn(processed_neg_rev, word)
        positive_score = positive_score + knn(processed_pos_rev, word)

    K_score = positive_score - negative_score

    if K_score >= 0:
        score.append('1')
    else:
        score.append('-1')

print("Writing to Format File...")
with open('format.txt', 'w', encoding="utf-8") as finalFile:
    finalFile.writelines("%s\n" % place for place in score)