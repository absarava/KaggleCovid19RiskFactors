import json
import glob
import re
import string
from random import sample

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from langdetect import detect
from langdetect import DetectorFactory
from sklearn.cluster import KMeans
import matplotlib.pyplot as py

import pandas as pd
import numpy as np

pd.set_option('display.max_colwidth', 1000)

# Each data source has a dataframe. Then we add a label to each that specifies what where it came from. For now we will use the metadata.csv file to analyze
col_list = ['cord_uid', 'sha', 'source_x', 'title', 'doi', 'pmcid', 'pubmed_id', 'license', 'abstract', 'publish_time',
            'authors', 'journal', 'Microsoft Academic Paper ID', 'WHO #Covidence', 'has_pdf_parse', 'has_pmc_xml_parse',
            'full_text_file', 'url']
metadata = pd.read_csv('C:/Users/arjun.b.saravanan/Desktop/CORD-19-research-challenge/metadata.csv', usecols=col_list)

# see how many papers don't have the abstract provided
# print(df['abstract'].isnull().sum())

# study basic info
# print(df.head())
# print(df.info())

# Loading JSON files
# Gathering path to all JSON files
# glob module allows for reading of all JSON files in the All JSON folder
all_json = glob.glob('C:/Users/arjun.b.saravanan/Desktop/CORD-19-research-challenge/All JSON/**/*.json',
                     recursive=True)
#print(metadata.loc[metadata['sha']=='5a6330a739f18fd6bc502bd9b59de55e8c081d4e'])

# Reader for all JSON files
class JSONFileReader:
    def __init__(self, file_path):
        with open(file_path) as file:
            content = json.load(file)
            # paper_id
            self.paper_id = content['paper_id']
            # title and initialize objects
            self.title = content['metadata'].get("title")
            self.abstract = []
            self.body_text = []
            # abstract
            for entry in content['abstract']:
                self.abstract.append(entry.get("text"))
            # body_text
            for entry in content['body_text']:
                self.body_text.append(entry.get("text"))
            self.abstract = '\n'.join(self.abstract)
            self.body_text = '\n'.join(self.body_text)
        def __repr__(self):
            return f'{self.paper_id}: {self.abstract[:200]}... {self.body_text[:200]}...'

#print(len(metadata))
#47324 records

datadict = {'paper_id': [], 'title': [], 'doi': [], 'journal': [], 'authors': [], 'abstract': [], 'body_text': []}
#not working at 4661
for entry in all_json[4500:4700]:
    try:
        reader = JSONFileReader(entry)
        #if sha is null go to metadata file and look for rest of details
        sub = metadata[metadata['sha'].str.contains(reader.paper_id, na = False, flags = re.IGNORECASE, regex = False)]
        datadict['paper_id'].append(reader.paper_id)
        datadict['title'].append(reader.title)
        doilist = list(sub['doi'])
        journallist = list(sub['journal'])
        #if len(doilist)==0: doilist.append("N/A")
        if not doilist:
            doilist.append('N/A')
        if not journallist:
            journallist.append('N/A')
        datadict['doi'].append(doilist[0]) #EXCEPTION
        datadict['journal'].append(journallist[0])
        authorlist = list(sub['authors'])
        if not authorlist:
            authorlist.append('N/A')
        datadict['authors'].append(authorlist[0])
        if not list(reader.abstract):
            reader.abstract = 'N/A'
        datadict['abstract'].append(reader.abstract)
        if not list(reader.body_text):
            reader.body_text = 'N/A'
        datadict['body_text'].append(reader.body_text)
        #if len(sub)==0:
         #   for x in datadict.keys():
          #      print(x, len(x))
           #     if len(datadict[x]) == 0:
            #        datadict[x] = "N/A"
    except Exception as e:
        continue

# 13039/47139 are null paper ids
trainingset = pd.DataFrame(datadict, columns = ['paper_id', 'title','doi','journal','authors','abstract','body_text'])
#print(trainingset.head())
#print(trainingset[trainingset.eq('N/A').any(1)])
#for x in datadict.keys():
    #print(x,datadict.get(x))
#nullcols = []
#for x in trainingset.columns:
 #  if sum(trainingset[x].isna())!=0:
#       nullcols.append(x)
#print(nullcols)

# Checked for duplicates. We can safely assume there are little to no duplicates in the JSON list and it is clean
#print(str(len(research)) + (' first length'))
#research.drop_duplicates()
#print(len(research))

# check for nulls in all columns
#null_columns = metadata.columns
#print(metadata[null_columns].isnull().sum())


# 13043 records in metadata do not have a paper_id to match to. So we should NOT match on paper_id.
# 186 records without a title, 3370 with no doi, 8277 with no abstract
# Need to account for this exception.
# If it doesn't have a paper_id in the metadata ('sha'), then match on doi
# print(trainingset.head())

# set seed
DetectorFactory.seed = 0

# hold label - language
langlist = []

# go through each text
for i in range(0, len(trainingset)):
    # split by space into list, take the first x intex, join with space
    text = trainingset.iloc[i]['body_text'].split(" ")

    lang = "en"
    try:
        if len(text) > 50:
            lang = detect(" ".join(text[:50]))
        elif len(text) > 0:
            lang = detect(" ".join(text[:len(text)]))
    except Exception as e:
        all_words = set(text)
        try:
            lang = detect(" ".join(all_words))
        # what!! :( let's see if we can find any text in abstract...
        except Exception as e:
            try:
                # let's try to label it through the abstract then
                lang = detect(trainingset.iloc[i]['abstract_summary'])
            except Exception as e:
                lang = "unknown"
                pass
    # get the language
    langlist.append(lang)

language_dict = {}
for x in langlist:
    language_dict[x] = langlist.count(x)

# graphic for languages
#py.bar(language_dict.keys(),language_dict.values(), color='r')
#py.show()

#not all sources are in english, lets do some more work understanding what languages are out there and use only English research
trainingset['languages'] = langlist
trainingset = trainingset[trainingset['languages']=='en']

# LET'S PROCESS THE TEXT
#Need to remove punctuation, make everything lowercase, add custom stopwords for word that could appear unnecessarily

#Remove punctuation
punctuations = string.punctuation
punc = list(punctuations)

custom_stop = ['``', ':', '/', 'N/A', 'etc.', 'it', 'The', 'For', ';', 'his', 'her', 'you', 'an', 'at', 'be', 'they', 'or', 'on', 'them', 'these', 'into', 'from', 'while', 'this', 'also', 'was', 'with', 'not', 'to', 'in', 'their', "''", 'are', 'by', 'per', 'as', 'is', 'that', 'the', 'and', 'of', 'a', 'for', 'doi', 'preprint', 'copyright', 'org', 'https', 'et', 'al', 'author', 'figure', 'table', 'http',
    'rights', 'I', 'biorxiv', 'medrxiv', 'license', 'fig', 'fig.', 'al.', 'Elsevier', 'PMC', 'CZI', '-PRON-', 'usually']
englishsw = set(stopwords.words('english'))
custom_stop.append(englishsw)

token_holder = []

for x in range(0, len(trainingset)):
    text = trainingset.iloc[x]['body_text']
    text_tokenized = word_tokenize(text)
    for i in range(0, len(text_tokenized)):
        text_tokenized[i] = text_tokenized[i].lower()
    text_tokenized = [word for word in text_tokenized if word not in custom_stop and word not in punc]
    token_holder.append(" ".join(text_tokenized))

token_final = pd.DataFrame(token_holder, columns=['body_text'])
#print(token_final)

#--------------------------------VECTORIZATION-----------------------------------

from sklearn.feature_extraction.text import TfidfVectorizer

# We need to vectorize the words to create values for analysis. How many words do we want to include in the TFidfVectorizer? Let's create a percentile.

def vectorize(text):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(text)
    return X

from sklearn.decomposition import PCA

vector = vectorize(token_final['body_text'].values)
arr = vector.toarray()
#Prints every corpus and its value for every word accordingly
#print(arr[:,0], arr[:,1])
#print(arr)
#print(arr.shape)
#print(vector)

#Because there are many features which may be collinear (causing multicollinearity issues), we are going to look into PCA

pca = PCA(n_components=.95)
reduced = pca.fit_transform(arr)
#print(reduced.shape)

#pca = PCA(n_components=.95)
#reduced = pca.fit_transform(arr)
#print(reduced.shape)


# Now we want to run our k means model for unsupervised learning
from sklearn.cluster import KMeans
#assigning k=20 as test
k = 20
kmeans = KMeans(n_clusters=20)
y_pred = kmeans.fit_predict(reduced)
token_final['kmean_fit'] = y_pred
#print(token_final['kmean_fit'])

from sklearn import metrics
from scipy.spatial.distance import cdist

# Let's find out how many ks we should use. This is done by minimizing the distortion,
# or sum of squared distances between each observation vector and the cluster centroid
distortionlist = []
K = range(1,50)
for k in K:
    kmean = KMeans(n_clusters=k).fit(reduced)
    kmean.fit(reduced)
    #Why do we have to fit it again?
    distortionlist.append(sum(np.min(cdist(reduced, kmean.cluster_centers_, 'euclidean'), axis=1)) / vector.shape[0])

#print(distortionlist)
#py.plot(K,distortionlist)
#py.show()

from sklearn.manifold import TSNE

tsne = TSNE()
tsne.fit(reduced)

vis_x = tsne[:, 0]
vis_y = tsne[:, 1]
py.scatter(vis_x, vis_y)
py.show()
#There are too many features to plot on a map. Each word in a TF-IDF vector is a feature, and each corpus, a sample. Sample size 'n' and feature size 'm' results in a lot of

#py.show()
#After finding a number for k, we want to see what that looks like on a plot

# RANDOM
# What are the features?? The WORDS are the features! Each word is a feature.

# The TFIDF Vectorizer should expect an array of strings. So if you pass it an array of arrays of tokens, it crashes.#
# So we want to add each corpus' words to a row in a dataframe
# Shape - first number is number of corpuses, second number is number of words total

# REMAINING ISSUES
# Need to handle possible duplicate articles
# Look into issue of file parsing
'''
cord_uid                          28
sha                            13043
source_x                          28
title                            186
doi                             3370
pmcid                          19288
pubmed_id                      11915
license                           28
abstract                        8277
publish_time                      37
authors                         2137
journal                         4433
Microsoft Academic Paper ID    46359
WHO #Covidence                 45555
has_pdf_parse                     29
has_pmc_xml_parse                 29
full_text_file                  8858
url                              331



resources: https://www.youtube.com/watch?v=sIyMAzAHw6I

'''



