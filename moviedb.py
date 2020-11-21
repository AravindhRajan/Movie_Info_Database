# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 14:24:47 2020

@author: Rajan
"""

import pandas as pd
import numpy as np
import streamlit as st
import imdb
import nltk
import spacy
from spacy.lang.en import English 
from spacy.lang.en.stop_words import STOP_WORDS #sw
from spacy.tokenizer import Tokenizer #tokenizer
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
#%matplotlib inline
from scipy.cluster.hierarchy import linkage, dendrogram
import pickle
import streamlit as st
#import omdb

moviesDB = imdb.IMDb()

# getting user input
movie_search = st.text_input('Enter the movie name here:') 

# searching for the movie
movies = moviesDB.search_movie(movie_search)

# preparing a filtered list based on the input
searchlist = []
for movie in movies:
    title = movie['title']
    year = movie['year']
    searchlist.append(f'{title} - {year}')
    #st.write(f'{title} - {year}')
    
#st.write('Movie chosen')
#st.write(movie_search)

#st.write('searchlist')
#st.write(type(searchlist))
#st.write(searchlist)

# asking the user to choose again from the fitlered list of movies
filtered_movie_search = st.selectbox('Choose the movie:',searchlist)

st.write('This is what you have chosen:')
st.write(filtered_movie_search)

# getting proper imdb name of the movie
if filtered_movie_search is not None:
    proper_movie = filtered_movie_search.split('-')[0].strip()

    #st.write('IMDB name of the movie:')
    #st.write(proper_movie)
          
    # getting imdb id of the chosen movie
    movies2 = moviesDB.search_movie(proper_movie)
    imdb_id = movies2[0].getID()
    
#    st.write('IMDB ID of the movie:')
#    st.write(imdb_id)
    
    # getting the moviename from imdb id
    search = moviesDB.get_movie(imdb_id) 
    #st.write(search)
    
    # getting release year of the movie
    year = search['year']
    
    # getting genre of the movie
    genre = []
    for i in search['genres']:
        genre.append(i)
    genre_all = ",".join(genre)
    
    # getting movie runtime
    runtime = []
    for i in search['runtimes']:
        runtime.append(i)
    runtime_all = runtime[0]+ " "+'mins'
    
    # getting movie certificate
    cert = []
    #[cert.append(i) for i in search['certificates']]
    for i in search['certificates']:
        cert.append(i)
    cert_india = [i for i in cert if 'India' in i][0] # filtering for india cert
    
    # getting movie langs
    lang = []
    #[lang.append(i) for i in search['languages']]
    for i in search['languages']:
        lang.append(i)
    lang_all = ",".join(lang)
    
    # getting the imdb rating
    try:
        rating = search.data['rating'] # picks summary written by users
#        st.write('IMDB rating of the movie:')
#        st.write(rating)
    except:
        st.write('Data not in IMDB')
        
    # consolidating basic info - movie
    st.write('Quick information about the movie')
    mov_detail = pd.DataFrame(columns=['Movie', 'Release Year', 'Imdb ID','Imdb Rating','Genre','Runtime','Certificate','Language(s)'])
    mov_detail = mov_detail.append({'Movie': proper_movie, 'Release Year': year, 'Imdb ID': imdb_id,'Imdb Rating':rating,'Genre':genre_all,'Runtime':runtime_all,'Certificate':cert_india,'Language(s)':lang_all}, ignore_index=True)
    st.write(mov_detail.transpose())
    # consolidating basic info - person
    #st.write('Quick information about the people')
    #person_detail = pd.DataFrame(columns=['Lead Actor','Role name(s) of the lead','Director'])
    #person_detail = person_detail.append({'Lead Actor':cast[0],'Role name(s) of the lead':role_all2,'Director':director[0]})
    #st.write(person_detail.transpose())
    
    # getting the lead actor from the movie
    cast = search['cast']
    st.write('Lead Actor:')
    st.write(cast[0]) 
    
    # getting the role name of the lead actor in the movie
    try:
        role = cast[0].currentRole
        st.write('Role name(s) of the lead actor:')
        for i in role:
            st.write(i)
            #print(i) 
            #role_all.append(i)
        #role_all2 = ",".join(role_all)    
    except:
        st.write('Data not in IMDB')

    # getting director name of the movie
    try:
        director = search['director']
        st.write('Director:')
        st.write(director[0]) 
    except:
        st.write('Data not in IMDB')
        
    # getting the movie plot
    try:
        plot = search['plot'] # picks summary written by users
        long_plot = max(plot, key=len) # pick the longest summary
        st.write('A short summary of the movie:')
        st.write(long_plot)
    except:
        st.write('Data not in IMDB')
    
    # getting similar movies
    nlp = spacy.load('en_core_web_sm', parser=False, entity=False) 
    df1 = pd.read_csv(r"F:\NIIT\Python\Py_prac\Plot_similarity\tamil_movies_formatted.csv")
    # custom sw list 
    custom_sw = ['justtrying']
    
    # Mark them as stop words
    for w in custom_sw:
        nlp.vocab[w].is_stop = True
        
        
    # new column with filtered, lemmatized words (did the belwo step and saved to the file)
#    df1['plot_filt_lem'] = df1.Plot.apply(lambda text: " ".join(token.lemma_ for token in nlp(text) 
#                                                       if not token.is_stop))
#    
#    df1.to_csv(r'F:\NIIT\Python\Py_prac\Plot_similarity\tamil_movies_formatted.csv')
    
    # create a tfidf vectorizer
    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                       min_df=0.2, stop_words='english',
                                       use_idf=True,ngram_range=(1,3))
        
    tfidf_matrix = tfidf_vectorizer.fit_transform([x for x in df1['plot_filt_lem']])

    # creating similarity matrix from cosing similarity
    similarity_distance = 1 - cosine_similarity(tfidf_matrix)
    
    st.write('Movies similar to the one you chose:')
    n = st.number_input('How many movies do you want',min_value=2,max_value=6)
 
    try:
        def suggest_sim_movies(proper_movie,n):
            index = df1[df1['Title'] == proper_movie].index[0]
            vector = similarity_distance[index, :]
            most_similar = df1.iloc[np.argsort(vector)[:n+1], 2].to_list()
            most_similar_list = most_similar[1:n+1]
            return most_similar_list
        
        final_lst = suggest_sim_movies(proper_movie,n)
        st.write('Movies similar to your selection:')
        st.write(final_lst)
    except:
        st.write('Data not enough')
        
    # getting the movie synopsis
#    try:
#        plot = search['synopsis'] # picks summary written by users
#        long_plot = max(plot, key=len) # pick the longest summary
#        st.write('A short synopsis of the movie:')
#        st.write(long_plot)
#    except:
#        st.write('Data not in IMDB')
#    
#    actor = moviesDB.get_person_infoset('0368400')
#    for job in actor['filmography'].keys():
#        print('# Job: ', job)
#        for movie in actor['filmography'][job]:
#            print('\t%s %s (role: %s)' % (movie.movieID, movie['title'], movie.currentRole))
#
#    persons = moviesDB.get_person(imdb_id) 
#    filmo = persons['filmography']
#    search = moviesDB.get_movie(imdb_id)
#    person = persons[0] 
#  
#    # getting more information 
#    moviesDB.update(person, info = ['filmography']) 
#    
#    # getting actor filmography
#    try:
#        filmography_list = []
#        actor_results = moviesDB.get_person_filmography(imdb_id)
#        for i in actor_results['data']['filmography']:
#            filmography_list.append(str(i))
#        st.write(filmography_list)
#    except:
#        st.write('Data not in IMDB')
    # printing person name from mmovie code
    #print(ia.get_person(code)) 
    
    # getting actor filmography 
    #actor_movies = moviesDB.get_person_filmography(imdb_id) 
    
    # printing movie name 
    #for i in range(5): 
    #   movie_name = actor_movies['data']['filmography'][0]['actor'][i]
    #  st.write(movie_name) 
    
    #latest_movie_name = actor_movies['data']['filmography'][0]['actor'][0]
    #st.write(latest_movie_name) 