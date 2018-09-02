from __future__ import print_function
import flask
from flask import Flask, render_template , request
from sklearn.externals import joblib
import numpy as np 
import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
from nltk.corpus import wordnet
from nltk import pos_tag

from sklearn.feature_extraction.text import CountVectorizer

import rake
import operator

import six

import mechanicalsoup
from bs4 import BeautifulSoup
import requests

app = Flask(__name__)
@app.route("/")
@app.route("/index")
def index():
       return flask.render_template('index.html')


def get_simple_pos(tag):
    
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
def clean_text(words):
    output_words = []
    stop = stopwords.words('english')
    stop += list(string.punctuation)
    lemmatizer = WordNetLemmatizer()
    for w in words:
        if w.lower() not in stop:
            pos = pos_tag([w])
            clean_word = lemmatizer.lemmatize(w, pos = get_simple_pos(pos[0][1]))
            output_words.append(clean_word.lower())
    return output_words
    
@app.route('/predict', methods=['POST'])
def make_prediction():
    if request.method=='POST':
        
        #get uploaded document
        file = request.files['uploaded_file']
        if not file: return render_template('index.html',label="No file uploaded")

        test =  pd.read_csv(file)
        line = list()
        for i in test[test.columns.tolist()]:
            line.append(i)
        
        X_test = ''
        X_test = X_test.join(line)

        text = word_tokenize(X_test)

        #text = ''.join(text)

        cleaned_text = clean_text(text)

        joined_text = " ".join(cleaned_text)

        rake_object = rake.Rake('stopwords.txt')
        sentenceList = rake.split_sentences(joined_text)
        stopwords = rake.load_stop_words('stopwords.txt')
        stopwordpattern = rake.build_stop_word_regex('stopwords.txt')
        phraseList = rake.generate_candidate_keywords(sentenceList, stopwordpattern, stopwords)
        wordscores = rake.calculate_word_scores(phraseList)
        keywordcandidates = rake.generate_candidate_keyword_scores(phraseList, wordscores)
        sortedKeywords = sorted(six.iteritems(keywordcandidates), key=operator.itemgetter(1), reverse=True)
        totalKeywords = len(sortedKeywords)
        for keyword in sortedKeywords[0:int(5)]:
            print("Keyword: ", keyword[0], ", score: ", keyword[1])


        x_test_mat = weight_model.transform(joined_text.split('.'))



        predict = model.predict(x_test_mat)
        print(predict)
        label = str(np.squeeze(predict[0]))
        print(label)
        #read_dict = np.load('final.npy').item()
        #id = read_dict[label]["doi"]
        #title = read_dict[label]["title"]
 
        #recommend = zip(id, title)

        browser = mechanicalsoup.StatefulBrowser()
        q = label
        browser.open("https://www.scimagojr.com/journalsearch.php?q="+q)
        soup = browser.get_current_page()
#print(soup.prettify())


#soup = BeautifulSoup(open("C:/Users/divya/Desktop/crawl.txt").read())
        divTag = soup.find_all("div", {"class" : "search_results"})
        l = len(divTag)
        divTag = str(divTag)
#print((divTag.split('</a>\n')[0]))

        recommend = []
        for i  in range(1,6):
            s = divTag.split('</a>\n')[i]
            a = s.split('>')
    #print(a)
            b = a[0].split('"')
            link = b[1]
            c = a[2].split('<')
    #print(c)
            title = c[0]
            recommend.append((title,"https://www.scimagojr.com/"+link))
            recommend = [(y,x.replace('amp;', '')) for y,x in recommend]
        print(recommend)

        return render_template('index.html', label=label, keyword=sortedKeywords[0:int(5)], recommendations=recommend)



if __name__ == '__main__':
    weight_model = joblib.load('weight_model.pkl')
    model = joblib.load('hacksvmmodel.pkl')
    app.run(host='127.0.0.1', port=8000, debug=True)