# Email Phishing Detection Project

## Overview
This project, developed as part of the NYU CS4CS (Computer Science for Cyber Security) program, explores the effectiveness of machine learning in detecting phishing emails compared to human detection capabilities.

## Project Description
The project implements and compares multiple machine learning models (Naive Bayes, Logistic Regression, and Support Vector Machine) to classify emails as either legitimate or phishing attempts. Using a curated dataset of both legitimate and phishing emails, the system aims to effectively identify malicious emails.

### Key Features
- Text preprocessing and vectorization using TF-IDF
- Implementation of three different ML models:
  - Naive Bayes
  - Logistic Regression
  - Support Vector Machine
- Cross-validation testing to ensure model reliability
- Comprehensive performance metrics including precision, recall, and F1-scores

## Dataset
The project utilizes a dataset of emails, split into training and testing sets, where each entry contains:
- Email subject
- Email body text
- Classification label (phishing/legitimate)

## Requirements
- Python 3.x
- scikit-learn
- pandas
- Jupyter Notebook

## Usage
The main analysis and code can be found in `code.ipynb`, which contains all the implementation details and results of the machine learning models.
