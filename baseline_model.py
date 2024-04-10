#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install simplet5

# pip install datasets


# In[6]:


from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from simplet5 import SimpleT5
from tabulate import tabulate
import nltk
from datetime import datetime
import numpy as np


# In[4]:


pip install simplet5


# In[5]:


pip install datasets


# In[7]:


import datasets
answersummData = datasets.load_dataset("alexfabbri/answersumm")


# In[8]:


answersummDataTest = answersummData["test"]
answersummDataTrain = answersummData["train"]
answersummDataVal = answersummData["validation"]


# In[9]:


answersummDataTest


# In[10]:


answersummDataTest[2]['question']['title']


# In[11]:


def flatten(example):
  answers = example["answers"]
#  print(answers[0])
  allanswers = ""
  for answer in answers:
    lines = answer["sents"]
    for line in lines:
      allanswers = allanswers + line["text"] + " "
    
  return {
      "title": example["question"]["title"],
      "question": example["question"]["question"],
      "allanswers": allanswers,
      "clustersummary": " ".join(example["cluster_summaries"][0]),
      "firstsummary": example["summaries"][0][0],
      "secondsummary": example["summaries"][0][1],
      "question+clustersumm": example["question"]["question"] + " " + " ".join(example["cluster_summaries"][0]),
      "question+allanswers": example["question"]["question"] + " " + allanswers
      }


# In[12]:


train_data_txt = answersummDataTrain.map(flatten, remove_columns=['answers', 'question', 'example_id', 'summaries', 'mismatch_info', 'annotator_id', 'cluster_summaries'])
val_data_txt = answersummDataVal.map(flatten, remove_columns=['answers', 'question', 'example_id', 'summaries', 'mismatch_info', 'annotator_id', 'cluster_summaries'])
test_data_txt = answersummDataTest.map(flatten, remove_columns=['answers', 'question', 'example_id', 'summaries', 'mismatch_info', 'annotator_id', 'cluster_summaries'])


# In[13]:


train_data_txt


# In[14]:


train_data_txt.column_names


# In[15]:


def average_token_len(data):
  sum = 0
  for i in range(len(data)):
    sum += len(data[i].split())
  return sum/len(data)


# In[16]:


print(average_token_len(train_data_txt['question']))
print(average_token_len(train_data_txt['allanswers']))
print(average_token_len(train_data_txt['firstsummary']))
print(average_token_len(train_data_txt['secondsummary']))
print(average_token_len(train_data_txt['clustersummary']))
print(average_token_len(train_data_txt['question+allanswers']))


# In[17]:


# Model-Bart trained on X-Sum

# model_name = "sshleifer/distilbart-xsum-12-3"
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)


# In[18]:


# Model T-5
model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-base')


# In[19]:


val_data_txt[0]["question+clustersumm"]


# In[20]:


"""### **Parameters**"""

encoder_max_length = 512
decoder_max_length = 128
input = "clustersummary"
output = "secondsummary"

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
    logging_steps=50,
    save_total_limit=3,
)


# In[21]:


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


# In[22]:


train_data = train_data_txt.map(
    lambda batch: batch_tokenize_preprocess(
        batch, tokenizer, encoder_max_length, decoder_max_length
    ),
    batched=True,
    remove_columns=train_data_txt.column_names,
)


# In[23]:


val_data = val_data_txt.map(
    lambda batch: batch_tokenize_preprocess(
        batch, tokenizer, encoder_max_length, decoder_max_length
    ),
    batched=True,
    remove_columns=val_data_txt.column_names,
)


# In[24]:


test_data = test_data_txt.map(
    lambda batch: batch_tokenize_preprocess(
        batch, tokenizer, encoder_max_length, decoder_max_length
    ),
    batched=True,
    remove_columns=test_data_txt.column_names,
)


# In[25]:


type(train_data_txt)


# In[26]:


pip install rouge_score


# In[27]:


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


# In[28]:


pip install sentencepiece


# In[29]:


pip install wandb


# In[30]:


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


# In[31]:


data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


# In[32]:


data_collator


# In[33]:


trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_data,
    eval_dataset=val_data,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)


# In[34]:


import os
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_NOTEBOOK_NAME"] = 't5model'


# In[35]:


#EVALUATE BEFORE FINE TUNING
trainer.evaluate()


# In[36]:


#TRAIN THE MODEL
trainer.train()


# In[37]:


#EVALUATE AFTER FINE TUNING
trainer.evaluate()


# In[38]:


#Generate summaries from the fine-tuned model and compare them with those generated from the original, pre-trained one.
def generate_summary(test_samples, model):
    inputs = tokenizer(
        test_samples['clustersummary'],
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


# In[39]:


pip install --upgrade transformers


# In[41]:


#model_before_tuning = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model_before_tuning = T5ForConditionalGeneration.from_pretrained('t5-small')
test_samples = val_data_txt.select(range(5))
model_after_training = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-xsum-12-3")
model_before_tuning = T5ForConditionalGeneration.from_pretrained('t5-small')
summaries_before_tuning = generate_summary(test_samples, model_before_tuning)[1]
summaries_after_tuning = generate_summary(test_samples, model)[1]
summaries_after_training = generate_summary(test_samples, model_after_training)[1]


# In[43]:


summaries_before_tuning


# In[44]:


summaries_after_tuning


# In[45]:


summaries_after_training


# In[46]:


for i in range(10):
  print(val_data_txt['secondsummary'][i])


# In[ ]:




