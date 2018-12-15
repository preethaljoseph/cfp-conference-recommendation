import json
from pandas.io.json import json_normalize
import pandas as pd
import gensim
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *
stemmer = PorterStemmer()
from urllib.request import urlopen
from bs4 import BeautifulSoup
from gensim import similarities
import numpy as np
np.random.seed(2018)

import nltk
nltk.download('wordnet')

output_json = json.load(open('./CRS_Data_Files/arxivData.json'))

# Print one line
print(output_json[1])

# Converts json tree structure to single line of text data
df = json_normalize(output_json)

print(df.head())

df["all_text"] = df["title"] + ". " + df["summary"]

# Converts all /n to spaces, lambda for looping
df["all_text"] = df["all_text"].map(lambda x: x.replace("\n", " "))
data_text = df["all_text"]
data_text['index'] = data_text.index
documents = data_text

print(df["all_text"].head(100))

def stemmLemm (text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(stemmLemm(token))
    return result


doc_sample = documents[1]
print(doc_sample)
words = []
for word in doc_sample.split(' '):
    words.append(word)
print(words)
print('\n\n tokenized and lemmatized document: ')
print(preprocess(doc_sample))

processed_docs = documents[1:10000].map(preprocess)
print(processed_docs)

dictionary = gensim.corpora.Dictionary(processed_docs)
count = 0
for k, v in dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break

dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
bow_corpus[4000]

bow_doc_4000 = bow_corpus[4000]
for i in range(len(bow_doc_4000)):
    print("Word {} (\"{}\") appears {} time.".format(bow_doc_4000[i][0], dictionary[bow_doc_4000[i][0]], bow_doc_4000[i][1]))

from gensim import corpora, models
tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]
from pprint import pprint
for doc in corpus_tfidf:
    pprint(doc)
    break

lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=10, id2word=dictionary, passes=2, workers=4)
for idx, topic in lda_model_tfidf.print_topics(-1):
    print('Topic: {} Word: {}'.format(idx, topic))

processed_docs[4000]


for index, score in sorted(lda_model_tfidf[bow_corpus[4000]], key=lambda tup: -1*tup[1]):
    print("\nScore: {}\t \nTopic: {}".format(score, lda_model_tfidf.print_topic(index, 10)))


# specify the url
quote_page = ["http://www.wikicfp.com/cfp/servlet/event.showcfp?eventid=82574&copyownerid=133374",
              "http://www.wikicfp.com/cfp/servlet/event.showcfp?eventid=67175&copyownerid=102557",
              "http://www.wikicfp.com/cfp/servlet/event.showcfp?eventid=83185&copyownerid=134892",
              "http://www.wikicfp.com/cfp/servlet/event.showcfp?eventid=81902&copyownerid=19391"
              ]

# query the website and return the html to the variable ‘page’
data = []
for pg in quote_page:
    page = urlopen(pg)

    # parse the html using beautiful soup and store in variable `soup`
    soup = BeautifulSoup(page, 'html.parser')

    # Take out the <div> of name and get its value
    descr_box = soup.find('div', attrs={'class': 'cfp'})
    description = descr_box.text.strip()  # strip() is used to remove starting and trailing

    conf_box = soup.find('span', attrs={'property': 'v:description'})
    conf_name = conf_box.text.strip()  # strip() is used to remove starting and trailing

    data.append((conf_name, description))
    # print(name)

print(data)
test_df = pd.DataFrame(data, columns=["conf_name", "description"])
unseen_document = test_df.loc[1,"conf_name"] + ' ' + test_df.loc[1,"description"]

print("This is the unseen document")

print(unseen_document)

bow_vector = dictionary.doc2bow(preprocess(unseen_document))
for index, score in sorted(lda_model_tfidf[bow_vector], key=lambda tup: -1*tup[1]):
    print("Score: {}\t Topic: {}".format(score, lda_model_tfidf.print_topic(index, 5)))



lda_index = similarities.MatrixSimilarity(lda_model_tfidf[corpus_tfidf])

# Let's perform some queries
similarities = lda_index[lda_model_tfidf[bow_vector]]
# Sort the similarities
similarities = sorted(enumerate(similarities), key=lambda item: -item[1])

# Top most similar documents:
print(similarities[:10])

# Let's see what's the most similar document
document_id, similarity = similarities[0]
print ("These are the document similarities ")
print(documents[document_id][:1000])


