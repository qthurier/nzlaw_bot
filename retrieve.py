import pandas as pd
import numpy as np
import gensim, logging
import numpy as np
import random
import scipy
from nltk.corpus import stopwords
import ast
import scipy
import math
import operator


stops = set(stopwords.words("english"))
model=gensim.models.KeyedVectors.load_word2vec_format('data/glove.6B.200d.w2c',binary=False) #GloVe Model
out_data_csv_path = 'data/QA_community_manual_with_answer_vector.csv' 
df = pd.read_csv(out_data_csv_path, sep='\t')


def get_glove_average(words):
    featureVec1 = np.zeros((200,), dtype="float32")
    nwords1 = 0

    for word in words.lower().split():
        if word not in stops:
            try:
                nwords1 = nwords1+1
                featureVec1 = np.add(featureVec1, model[word])
            except:
                pass
            
    if(nwords1>0):
            featureVec1 = np.divide(featureVec1, nwords1)

    return featureVec1



def glove_average(row):
    return get_glove_average(row["question"])


def get_similar_questions_with_topic(question, topic=18):
    results = []
    index_q = get_glove_average(question)
    same_topic = df["topic_number"] == topic
    instance_same_topic = df.loc[same_topic]
    for index, row in instance_same_topic.iterrows():
        item_question = row["question"]
        item_vector = row["6b_glove_avg_200"]
        item_answer = row["answer"]
        item_vector = ast.literal_eval(item_vector)
        similarity_score =  1 - scipy.spatial.distance.cosine(item_vector,index_q)
        results.append((item_question,similarity_score,item_answer))
    best = sorted(results, key=lambda x: x[1], reverse=True)
    for item in best:
        if math.isnan(item[1]):
            pass
        else:
            print("%s\t%s" % (item[1],item[0]))
    return best[0][2]



def get_similar_questions(question):
    results = pd.DataFrame(columns=['topic_number', 'score', 'question','answer'])
    index_q = get_glove_average(question)
    for index, row in df.iterrows():
        item_question = row["question"]
        item_vector = row["6b_glove_avg_200"]
        item_topic_number = row["topic_number"]
        item_answer = row["answer"]
        item_vector = ast.literal_eval(item_vector)
        similarity_score =  1 - scipy.spatial.distance.cosine(item_vector,index_q)
        #results.append((item_topic_number,item_question,similarity_score))
        results.loc[-1] = [item_topic_number, similarity_score, item_question,item_answer]  # adding a row
        results.index = results.index + 1
    results = results.sort_values('score', ascending=False)
    return results


def get_similar_questions_answers(question,topic=0,q_weight=0.3,a_weight=0.7):
    if topic!=0:
        same_topic = df["topic_number"] == topic
        instance_same_topic = df.loc[same_topic]
    else:
        instance_same_topic = df
    
    results = pd.DataFrame(columns=['topic_number', 'score', 'question','answer'])
    index_q = get_glove_average(question)
    for index, row in instance_same_topic.iterrows():
        item_question = row["question"]
        item_vector = row["6b_glove_avg_200"]
        item_vector = ast.literal_eval(item_vector)
        item_vector_answer = row["6b_glove_avg_200_answer"]
        item_vector_answer = ast.literal_eval(item_vector_answer)        
        item_topic_number = row["topic_number"]
        item_answer = row["answer"]
        
        similarity_score_q =  1 - scipy.spatial.distance.cosine(item_vector,index_q)
        similarity_score_a =  1 - scipy.spatial.distance.cosine(item_vector_answer,index_q)
        final_score = q_weight * similarity_score_q + a_weight * similarity_score_a
        results.loc[-1] = [item_topic_number, final_score, item_question,item_answer]  # adding a row
        results.index = results.index + 1
    results = results.sort_values('score', ascending=False)
    print(results[['score', 'question']].head()) # for debugging purposes
    return results[['score', 'question', 'answer']].head().reset_index() # results['answer'].iloc[0]


