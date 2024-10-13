Phishing Detection Using Large Language Models and Machine Learning
This repository contains the code and notebooks for a project focused on comparing different machine learning, deep learning, and large language models for phishing email detection. The study evaluates models such as Random Forest, KNN, SVM, CNN, LSTM, RoBERTa, BERT, DistilBERT, and the Mistral 7B model, with the goal of improving the detection of phishing emails and distinguishing them from non-phishing emails and spam.

Project Structure
mistral-focus-on-binary-training.ipynb: This notebook fine-tunes the Mistral 7B model for phishing detection using binary classification. It focuses on optimizing the model’s performance through iterative prompt engineering and memory-efficient methods.

phishing-detection.ipynb: This notebook implements traditional machine learning algorithms (Random Forest, KNN, SVM) and deep learning models (CNN, LSTM) to detect phishing emails. It includes preprocessing steps such as text cleaning, tokenization, and vectorization.

app.py: This file contains the code for deploying the Mistral model using Hugging Face Spaces and Gradio. The model classifies email content as either phishing (1) or non-phishing (0) based on user input.

Datasets
The project makes use of multiple datasets for training and evaluation:

TREC 07 Public Corpus: A preprocessed version of the TREC 2007 dataset containing spam and ham emails. This dataset is used to test the generalizability of the phishing detection models on spam.

Source: Kaggle: Preprocessed TREC 2007 Public Corpus Dataset
Nazario Dataset: A curated dataset of phishing emails. This dataset contains phishing emails from the publicly available Nazario dataset, combined with ham emails from other sources such as SpamAssassin, TREC, and CEAS. It was used for training and evaluating the phishing detection models.

How to Run the Notebooks
Prerequisites
Python 3.7 or higher

Install the required dependencies by running:

pip install -r requirements.txt
Make sure you have access to the required datasets. You can download the TREC_07 dataset from Kaggle.

Running the Notebooks
Mistral 7B Model Fine-Tuning:

Open mistral-focus-on-binary-training.ipynb.
Ensure that you have access to a GPU for faster training, or modify the code to run on a CPU (although this will be slower).
Follow the steps in the notebook to fine-tune the Mistral model on the phishing dataset.
Machine Learning and Deep Learning Models:

Open phishing-detection.ipynb.
Preprocess the datasets as shown in the notebook and train models such as Random Forest, KNN, CNN, and LSTM.
Evaluate the models' accuracy, precision, and recall on both phishing and spam datasets.
Deployment
To deploy the Mistral model using Hugging Face Spaces:

Upload the app.py file to a Hugging Face Space.
Ensure the Hugging Face library and dependencies (like Gradio) are installed.
Run the app.py file. This will launch a Gradio interface where users can input email content to be classified as phishing or non-phishing by the model.
Model Performance
The study found that the Mistral 7B model, with fine-tuned prompt engineering, achieved the best performance with an accuracy of 96.345% on the phishing dataset and 92.123% on the spam dataset. Other models such as RoBERTa and CNN also performed well but were outperformed by the Mistral model, especially after the introduction of explicit spam classification.

License
This project is licensed under the MIT License.