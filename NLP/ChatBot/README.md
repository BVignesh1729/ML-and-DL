## CHATBOT

A chatbot, which is built upon preprocessed input(preprocessing involves stemming and vectorization, using the bag of words paradigm).
Works on a simple Neural Net with 2 layers. The bot's training data is a .json file and is customizable, and can be changed according to the requirement.

Preprocessing :
Every query is initially tokenized, after which stemming is applied in order to avoid overfitting to the same word in different forms(in terms on tense, etc.). In a way, it "normalizes" the data.


