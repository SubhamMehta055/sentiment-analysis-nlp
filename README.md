# Sentiment Analysis on Amazon Fine Food Reviews

## Overview
This project is an end-to-end sentiment analysis system built to classify customer reviews as positive or negative using natural language processing (NLP) techniques. The aim is to gain insights into customer satisfaction, which can help businesses enhance their products and services. This system compares multiple models to achieve high accuracy and is integrated with a Flask web application for ease of use.

## Introduction
In the competitive food industry, understanding customer sentiment is crucial for improving product offerings and customer satisfaction. Analyzing customer reviews can provide valuable insights into customer opinions. This project implements sentiment analysis using the Amazon Fine Food Reviews dataset from Kaggle. We leverage traditional and deep learning approaches to classify the reviews, providing a robust sentiment classification system.

## Understanding Sentiment Analysis
Sentiment analysis, a branch of NLP, focuses on determining the emotional tone behind textual content. There are several methods for performing sentiment analysis, including:

- **Rule-Based Approaches**: Use predefined dictionaries or lexicons for scoring text.
- **Machine Learning Models**: Train traditional or deep learning models to predict sentiment.
- **Pretrained Transformer Models**: Utilize state-of-the-art NLP models, like RoBERTa, for contextual understanding and high accuracy.

## Building the Sentiment Analysis System
The project involves collecting and preprocessing text data, followed by implementing several sentiment analysis models:

1. **VADER (Valence Aware Dictionary and sEntiment Reasoner)**: A rule-based model using a bag-of-words approach for quick sentiment scoring.
2. **Logistic Regression with Cross-Validation**: A traditional machine learning model used as a baseline, achieving an accuracy of 72%.
3. **RoBERTa (Pretrained Transformer Model)**: A high-performance model that achieved an accuracy of 95%, leveraging deep contextual understanding of text for precise sentiment classification.

## Dataset Description
The Amazon Fine Food Reviews dataset includes:

- **Reviews**: Textual data detailing customer opinions on various food products.
- **Ratings**: Scores indicating the user's satisfaction level.

The dataset is preprocessed to handle missing values, remove noise, and prepare for model training and evaluation.

## Integrating with Flask
After building the sentiment analysis system, it was integrated with a Flask web application to provide an interactive user interface. The Flask application allows users to input their own text or select from sample reviews and view the sentiment prediction in real time. This enables users to explore the model’s performance and understand customer sentiments directly.

## Project Highlights
- **Multiple Model Implementations**:
  - VADER for quick scoring with a rule-based approach.
  - Logistic Regression CV for a traditional baseline comparison.
  - RoBERTa, which provided the best performance, for deep learning-based sentiment analysis.
  
- **Flask Integration**: A streamlined web application for user-friendly interaction with the sentiment analysis system.

## Key Functionalities

### Data Collection & Preprocessing
- Text data cleaning, tokenization, and normalization.
- Handled missing values, removed unnecessary symbols, and converted text to lower case.

### Sentiment Analysis Models
- **VADER**: Quick and effective for short texts; uses a bag-of-words lexicon-based scoring.
- **Logistic Regression CV**: A traditional model used for baseline sentiment prediction.
- **RoBERTa**: Transformer model providing high accuracy by leveraging deep contextual learning.

## Achievements
- **High Accuracy with RoBERTa**: Achieved a robust accuracy of 95% using RoBERTa for sentiment classification.
- **End-to-End Development**: Successfully built a complete system from data preprocessing, model training, and evaluation to deployment within a Flask application.
- **Real-World Impact**: Provides valuable insights for understanding customer sentiment in the food industry, helping businesses improve products and customer satisfaction.

## Business Benefits
- **Improved Customer Insights**: Allows businesses to understand customer sentiment at scale, helping to address common issues and preferences.
- **Informed Decision-Making**: Sentiment trends can guide product development and marketing strategies.
- **Competitive Edge**: Leveraging customer feedback for continuous improvement increases customer satisfaction and loyalty.

## Conclusion
This project demonstrates the potential of NLP and machine learning in understanding customer sentiment from textual data. By integrating with a Flask web application, this system offers a user-friendly platform for real-time sentiment analysis, allowing businesses to gain actionable insights directly from customer reviews. The combination of VADER, Logistic Regression, and RoBERTa ensures a comprehensive approach to sentiment analysis, making this system a valuable tool for any organization aiming to enhance customer satisfaction.

## Technologies Used
- Python
- Flask
- VADER
- Hugging Face Transformers
- RoBERTa
- Pandas
- NLTK
- Kaggle (Amazon Fine Food Reviews dataset)
