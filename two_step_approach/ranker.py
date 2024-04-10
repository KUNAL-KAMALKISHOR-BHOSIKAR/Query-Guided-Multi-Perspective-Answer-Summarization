#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install transformers')
get_ipython().system('pip install datasets')
get_ipython().system('pip install sentencepiece')
get_ipython().system('pip install rouge_score')
get_ipython().system('pip install simplet5 -q')


# In[2]:


get_ipython().system('pip install rank_bm25')


# In[3]:


get_ipython().system('pip install sentence-transformers')


# In[4]:


import json
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import torch
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction import _stop_words
import string
from tqdm.autonotebook import tqdm
import numpy as np
import pickle


# In[5]:


import datasets
answersummData = datasets.load_dataset("alexfabbri/answersumm")


# In[6]:


answersummData


# In[7]:


Test_data = answersummData["test"]
Train_data = answersummData["train"]
Val_data = answersummData["validation"]


# In[8]:


def flatten(example):
  answers = example["answers"]
#  print(answers[0])
  allanswers = []
  for answer in answers:
    lines = answer["sents"]
    for line in lines:
      allanswers.append(line["text"])
    
  return {
      "question": example["question"]["question"],
      "allanswers": allanswers,
      "firstsummary": example["summaries"][0][0],
      }


# In[9]:


Train_data_txt = Train_data.map(flatten, remove_columns=['answers', 'question', 'example_id', 'summaries', 'mismatch_info', 'annotator_id', 'cluster_summaries'])
Val_data_txt = Val_data.map(flatten, remove_columns=['answers', 'question', 'example_id', 'summaries', 'mismatch_info', 'annotator_id', 'cluster_summaries'])
Test_data_txt = Test_data.map(flatten, remove_columns=['answers', 'question', 'example_id', 'summaries', 'mismatch_info', 'annotator_id', 'cluster_summaries'])


# In[10]:


"""### **BM25**"""

# We lower case our text and remove stop-words from indexing
def bm25_tokenizer(text):
    tokenized_doc = []
    for token in text.lower().split():
        token = token.strip(string.punctuation)

        if len(token) > 0 and token not in _stop_words.ENGLISH_STOP_WORDS:
            tokenized_doc.append(token)
    return tokenized_doc

def answers_tokenized(allanswers):
  tokenized_corpus = []
  for answer in tqdm(allanswers):
    tokenized_corpus.append(bm25_tokenizer(answer))
  return tokenized_corpus


# In[11]:


i = 5
query = Val_data_txt[i]["question"]
tokenized_corpus = answers_tokenized(Val_data_txt[i]["allanswers"])
bm25 = BM25Okapi(tokenized_corpus)
bm25_scores = bm25.get_scores(bm25_tokenizer(query))
ranked_ans = bm25.get_top_n(bm25_tokenizer(query), Val_data_txt[i]["allanswers"], n=20)

ranked_ans

Val_data_txt[5]["allanswers"]


# In[12]:


def ranked_ans_bm25(corpus):
  newcorpus = []
  for i in range(len(corpus)):
    newcorpus_dict = {}
    query = corpus[i]["question"]
    tokenized_corpus = answers_tokenized(corpus[i]["allanswers"])
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(bm25_tokenizer(query))
    ranked_ans = bm25.get_top_n(bm25_tokenizer(query), corpus[i]["allanswers"], n=20)
    #print('ranked answer ', ranked_ans)
    newcorpus_dict["question"] = query
    newcorpus_dict["rankedanswers"] = ranked_ans
    newcorpus_dict["firstsummary"] = corpus[i]["firstsummary"]
    newcorpus.append(newcorpus_dict)
#    print('corpus rankedanswers ', corpus[i]["rankedanswers"])
  return newcorpus


# In[13]:


Train_bm25 = ranked_ans_bm25(Train_data_txt)
Val_bm25 = ranked_ans_bm25(Val_data_txt)
Test_bm25 = ranked_ans_bm25(Test_data_txt)


# In[14]:


pickle_out = open("Train-bm25.pickle","wb")
pickle.dump(Train_bm25, pickle_out)
pickle_out.close()


# In[15]:


"""## **Bi-Encoder**"""


# In[16]:


bi_encoder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
bi_encoder.max_seq_length = 256     #Truncate long passages to 256 tokens

top_k = 20


# In[17]:


def ranked_ans_bi_encoder(corpus):
  newcorpus = []
  for i in range(len(corpus)):
    newcorpus_dict={}
    query = corpus[i]["question"]
    question_embedding = bi_encoder.encode(query, convert_to_tensor=True)
    question_embedding = question_embedding.cuda()
    corpus_embeddings = bi_encoder.encode(corpus[i]["allanswers"], convert_to_tensor=True)
    scores = util.dot_score(question_embedding, corpus_embeddings)[0].cuda().tolist()
    doc_score_pairs = list(zip(corpus[i]["allanswers"], scores))
    doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
    i = 0
    ranked_ans = []
    for doc, score in doc_score_pairs:
      if i<20:
        ranked_ans.append(doc)
      i +=1
    #print(len(ranked_ans))    
    #hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=top_k)
    #print(hits)
    #ranked_ans = sorted(corpus[i]["allanswers"], key=lambda x: x['score'], reverse=True)
    newcorpus_dict["question"] = query
    newcorpus_dict["rankedanswers"] = ranked_ans
    newcorpus_dict["firstsummary"] = corpus[i]["firstsummary"]
    newcorpus.append(newcorpus_dict)
    #corpus[i]["allanswers"] = ranked_ans
  return newcorpus


# In[18]:


Train_bi_encoder = ranked_ans_bi_encoder(Train_data_txt)
Val_bi_encoder = ranked_ans_bi_encoder(Val_data_txt)
Test_bi_encoder = ranked_ans_bi_encoder(Test_data_txt)


# In[19]:


pickle_out = open("Train-bi-encoder.pickle","wb")
pickle.dump(Train_bi_encoder, pickle_out)
pickle_out.close()




# In[20]:


pickle_out = open("Val-bi-encoder.pickle","wb")
pickle.dump(Val_bi_encoder, pickle_out)
pickle_out.close()



# In[21]:


pickle_out = open("Test-bi-encoder.pickle","wb")
pickle.dump(Test_bi_encoder, pickle_out)
pickle_out.close()


# In[22]:


"""### **Cross-Encoder**"""


# In[23]:


cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')


# In[24]:


def ranked_ans_cross_encoder(corpus):
  newcorpus = []
  for i in range(len(corpus)):
    newcorpus_dict={}
    query = corpus[i]["question"]
    cross_inp = [[query, answer] for answer in corpus[i]['allanswers']]
    cross_scores = cross_encoder.predict(cross_inp)
    doc_score_pairs = list(zip(corpus[i]["allanswers"], cross_scores))
    doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
    i = 0
    ranked_ans = []
    for doc, score in doc_score_pairs:
      if i<20:
        ranked_ans.append(doc)
      i +=1
    #print(ranked_ans)
    #print(len(ranked_ans)) 
    #break 
    newcorpus_dict["question"] = query
    newcorpus_dict["rankedanswers"] = ranked_ans
    newcorpus_dict["firstsummary"] = corpus[i]["firstsummary"]
    newcorpus.append(newcorpus_dict)
  return newcorpus


# In[25]:


from sentence_transformers import SentenceTransformer, CrossEncoder, util


# In[26]:


Train_cross_encoder = ranked_ans_cross_encoder(Train_data_txt)
Val_cross_encoder = ranked_ans_cross_encoder(Val_data_txt)
Test_cross_encoder = ranked_ans_cross_encoder(Test_data_txt)


# In[27]:


# Write Ranked-Cross-encoder into pickle file


# In[28]:


pickle_out = open("Train-cross-encoder.pickle","wb")
pickle.dump(Train_cross_encoder, pickle_out)
pickle_out.close()


# In[29]:


pickle_out = open("Val-cross-encoder.pickle","wb")
pickle.dump(Val_cross_encoder, pickle_out)
pickle_out.close()


# In[30]:


pickle_out = open("Test-cross-encoder.pickle","wb")
pickle.dump(Test_cross_encoder, pickle_out)
pickle_out.close()


# In[ ]:




