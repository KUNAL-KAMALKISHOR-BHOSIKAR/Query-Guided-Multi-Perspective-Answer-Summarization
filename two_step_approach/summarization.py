#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install transformers')
get_ipython().system('pip install datasets')
get_ipython().system('pip install sentencepiece')
get_ipython().system('pip install rouge_score')
get_ipython().system('pip install simplet5 -q')


# In[2]:


from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from simplet5 import SimpleT5
from tabulate import tabulate
from datasets import Dataset
import nltk
from datetime import datetime
import numpy as np
import pickle
import datasets


# In[3]:


import os
os.environ["WANDB_DISABLED"] = "true"


# In[4]:


Train = open("/kaggle/input/rankar-data/Ranked-data/Train-cross-encoder.pickle","rb")
Train_data = pickle.load(Train)


# In[ ]:





# In[5]:


Val = open("/kaggle/input/rankar-data/Ranked-data/Val-cross-encoder.pickle","rb")
Val_data = pickle.load(Val)


# In[6]:


Test = open("/kaggle/input/rankar-data/Ranked-data/Test-cross-encoder.pickle","rb")
Test_data = pickle.load(Test)


# In[7]:


import pandas as pd
import datasets

# Assuming Train_data, Val_data, and Test_data are lists of dictionaries
train_df = pd.DataFrame(Train_data)
val_df = pd.DataFrame(Val_data)
test_df = pd.DataFrame(Test_data)


# In[8]:


# Convert Pandas DataFrames to datasets
train_data_txt = datasets.Dataset.from_pandas(train_df)
val_data_txt = datasets.Dataset.from_pandas(val_df)
test_data_txt = datasets.Dataset.from_pandas(test_df)


# In[9]:


def flatten(example):
  answers = example["rankedanswers"]
#  print(answers[0])
  allrankedanswers = ""
  for answer in answers:
    allrankedanswers += answer + " "
    
  return {
      
      "question": example["question"],
      "allrankedanswers": allrankedanswers,
      "firstsummary": example["firstsummary"],
      "question+allrankedanswers": example["question"] + " " + allrankedanswers
      }


# In[10]:


train_data_txt = train_data_txt.map(flatten, remove_columns=['rankedanswers'])
val_data_txt = val_data_txt.map(flatten, remove_columns=['rankedanswers'])
test_data_txt = test_data_txt.map(flatten, remove_columns=['rankedanswers'])


# In[11]:


len(val_data_txt)


# In[12]:


val_data_txt[0]


# In[13]:


# Model-Bart trained on X-Sum

model_name = "sshleifer/distilbart-xsum-12-3"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


# In[14]:


# Model T-5
#model = T5ForConditionalGeneration.from_pretrained('t5-small')
#tokenizer = T5Tokenizer.from_pretrained('t5-base')


# In[27]:


encoder_max_length = 512
decoder_max_length = 100
input = "allrankedanswers"
output = "firstsummary"

train_data_txt.column_names

training_args = Seq2SeqTrainingArguments(
    output_dir="results",
    num_train_epochs=5,  # demo
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=2,  # demo
    per_device_eval_batch_size=2,
    # learning_rate=3e-05,
    warmup_steps=100,
    weight_decay=0.1,
    label_smoothing_factor=0.1,
    predict_with_generate=True,
    logging_dir="logs",
    logging_steps=200,    ## it take lot of output space  so i incresead logging step 50 to 200
    save_total_limit=1,
)


# In[28]:


def batch_tokenize_preprocess(batch, tokenizer, max_source_length, max_target_length):
    source, target = batch[input], batch[output]
    source_tokenized = tokenizer(
        source, padding="max_length", truncation=True, max_length=max_source_length
    )
    target_tokenized = tokenizer(
        target, padding="max_length", truncation=True, max_length=max_target_length
    )

    batch = {k: v for k, v in source_tokenized.items()}
    # Ignore padding in the loss
    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in l]
        for l in target_tokenized["input_ids"]
    ]
    return batch


# In[29]:


train_data = train_data_txt.map(
    lambda batch: batch_tokenize_preprocess(
        batch, tokenizer, encoder_max_length, decoder_max_length
    ),
    batched=True,
    remove_columns=train_data_txt.column_names,
)


# In[30]:


val_data = val_data_txt.map(
    lambda batch: batch_tokenize_preprocess(
        batch, tokenizer, encoder_max_length, decoder_max_length
    ),
    batched=True,
    remove_columns=val_data_txt.column_names,
)


# In[31]:


test_data = test_data_txt.map(
    lambda batch: batch_tokenize_preprocess(
        batch, tokenizer, encoder_max_length, decoder_max_length
    ),
    batched=True,
    remove_columns=test_data_txt.column_names,
)


# In[32]:


#METRICS
nltk.download("punkt", quiet=True)

metric = datasets.load_metric("rouge")

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


# In[33]:


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    # Extract a few results from ROUGE
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
    ]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


# In[34]:


data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


# In[35]:


trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_data,
    eval_dataset=val_data,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)


# In[36]:


#EVALUATE BEFORE FINE TUNING
trainer.evaluate()


# In[37]:


#TRAIN THE MODEL
trainer.train()


# In[38]:


#EVALUATE AFTER FINE TUNING
trainer.evaluate()


# In[40]:


#Generate summaries from the fine-tuned model and compare them with those generated from the original, pre-trained one.
def generate_summary(test_samples, model):
    inputs = tokenizer(
        test_samples["allrankedanswers"],
        padding="max_length",
        truncation=True,
        max_length=encoder_max_length,
        return_tensors="pt",
    )
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)
    outputs = model.generate(input_ids, attention_mask=attention_mask)
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return outputs, output_str


# In[41]:


model_before_tuning = AutoModelForSeq2SeqLM.from_pretrained(model_name)
#model_before_tuning = T5ForConditionalGeneration.from_pretrained('t5-small')
test_samples = val_data_txt.select(range(5))
model_after_training = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-xsum-12-3")
summaries_before_tuning = generate_summary(test_samples, model_before_tuning)[1]
summaries_after_tuning = generate_summary(test_samples, model)[1]
summaries_after_training = generate_summary(test_samples, model_after_training)[1]


# In[45]:


for i in range(5):
  print(val_data_txt['firstsummary'][i])

# print(summaries_before_tuning)

# print(summaries_after_tuning)


# In[43]:


print(summaries_before_tuning)


# In[44]:


print(summaries_after_tuning)


# In[ ]:




