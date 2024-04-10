# Query-Guided-Multi-Perspective-Answer-Summarization

Our submission contains two folders: notebooks and two_step_approach including baseline_model and report .

notebooks : this folder contains implemented all our code which was implemented in kaggle notebooks .
two_step_approach : this folder contains code and reanked data .



To install all the requirements, run the following command:

pip install -r requirements.txt
Note: We originally implemented all our codes on kaggle notebooks.

Baseline Model Instructions:-
-----------------------------------------------------------------
Running the Baseline Model
Change the Model Type:
Navigate to the MODEL section in baseline_model.py.
To use the T5 model: uncomment the codes under model-T5 and comment out the codes under model-Bart.
To use the Bart model: uncomment the codes under model-bart and comment out the codes under model-T5.
Change Source, Source Length, Target, Target Length:



Two-Step Approach:-
----------------------------------------------


Run ranker.py:
It ranks the answers and creates a new dataset using the top-20 ranked answers.
Parameters:
Source length = encoder_max_length
Target length = decoder_max_length
Input options:
"allanswers"
"question+allanswers"
"clustersummary"
"question+clustersummary"
Target options:
"firstsummary" (if using answers in input)
"secondsummary" (if using clustersummary in input)
Data Folder:

The Ranked-data Folder already contains the ranked-data created by BM25, Bi-encoder, Cross-encoder.
Running the Summarization:

Then run summarization.py:
It evaluates the model on the ranked data. The code is similar to baseline_model.py.




For a detailed explanation and demonstration,  you can see presentation
