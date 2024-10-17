# AAI-520-group-04
Project Title: Building a Chatbot using the Cornell Movie Dialogs Corpus
Short Description and Objectives:
This project aims to develop a multi-turn conversational chatbot using the Cornell Movie-Dialogs Corpus and transformer-based models like BERT or GPT. The process involves analyzing and preprocessing the corpus, training and fine-tuning the model for multi-turn responses, and evaluating performance using metrics like BLEU score and perplexity. The final step is to deploy the chatbot for interactive testing and feedback to enhance its conversational capabilities.  
Dataset Description:
The Cornell Movie-Dialogs Corpus is a rich dataset collected from publicly available movie scripts, with metadata enhanced through IMDb. It contains:
220,579 conversational exchanges between 10,292 pairs of movie characters.
304,713 individual lines of dialogue (utterances).
617 movies, each enriched with detailed metadata like genres, release year, IMDb ratings, and number of votes.
9,035 unique characters, with information on gender and credit position for a subset of 3,774 characters.
Project Phases:
1. Research and Study Phase:
Study generative-based chatbot architectures like Seq2Seq, Transformers, and GPT.
Understand the challenges of chatbot design: context management, coherency, handling ambiguous queries, etc.
2. Data Collection and Preprocessing:
Dataset: Cornell Movie Dialogs Corpus.
Preprocessing Steps:
Tokenization: Splitting text into tokens.
Handling different languages (if applicable).
Managing context: Grouping utterances into conversations.
3. Model Design and Training:
Architecture: Transformer-based models (e.g., GPT-2).
Implementation: Use existing implementations to train the model with the dataset.
4. Evaluation:
Metrics: BLEU score, Perplexity.
Evaluation: Assess the model's ability to carry out multi-turn conversations, adapt to context, and handle a variety of topics.
Implementation in Jupyter Notebook: