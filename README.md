# GPT2IdentityInfilling

This ReadMe will define the steps that are required to run the experiment

1. packages required
  a. transformers
  b. torch
  c. nltk
  d. pandas
2. Install the packages, after successful installation go to the python/libs/site-packages/ transformers/src/transformers/models/gpt2/gpt2_modeling.py
3. Replace the line in GPT2LMHeadModel named loss = nn.CrossEntropy(logits, labels) to loss = nn.CrossEntropy(logits, labels, ignore_index = -1)
4. After successful replacement, run the infill_gpt2.py using >> python infill_gpt2.py --epochs <num_epochs>
5. After that run the Testing_GPT2.ipynb file
