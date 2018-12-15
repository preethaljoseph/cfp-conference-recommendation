# cfp-conference-recommendation
Conference Recommendation System

The following dataset needs to downloaded and saved to the project location. I am unable to upload to the repo as it is above 25MB: 

1.	Arxiv dataset from Kaggle.com - https://www.kaggle.com/neelshah18/arxivdataset - This dataset contains information about the authors, summary of paper, title and tag. 

In this project we will use topic modelling and document similarity to recommend conferences to users based on the work they have published previously. We use two datasets for this project: 
1.	Arxiv dataset from Kaggle.com - https://www.kaggle.com/neelshah18/arxivdataset - This dataset contains information about the authors, summary of paper, title and tag. 
2.	Conference details dataset scraped from the website http://www.wikicfp.com/cfp/ - For information on conference Call for Papers and conference names. 

Some of the techniques implemented in this project include: 
1.	TF-IDF Transformation
2.	LDA Topic Modelling 
3.	Document Similarity

Setup: 

# install pandas
pip install pandas

# install genism
pip install –upgrade genism

# install nltk
sudo pip install -U nltk

#install numpy
sudo pip install -U numpy

#install BeautifulSoup 
pip install beautifulsoup4

#install urllib3
pip install urllib3

Start: 

Let us start by importing the following libraries: 

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

