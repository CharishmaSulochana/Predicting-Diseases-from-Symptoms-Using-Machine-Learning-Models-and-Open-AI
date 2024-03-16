# Predicting-Diseases-from-Symptoms-Using-Machine-Learning-Models-and-Open-AI


# Problem Statement:
Delayed or neglected medical attention for minor symptoms is a common challenge, leading to potential complications and adverse health outcomes. There is a need for a solution that can enhance user-centric healthcare by accurately predicting diseases based on symptoms, facilitating timely medical intervention and knowledge dissemination.

# Abstract:
This project aims to address the challenge of delayed or neglected medical attention for minor symptoms by developing a machine learning model integrated with OpenAI technologies for accurate disease prediction and knowledge dissemination. The project focuses on illness prediction based on symptoms provided by the user, utilizing a powerful machine learning model and OpenAI technologies. The dataset consists of 407 columns containing symptoms and 135 rows containing disease names, with each column representing a symptom with binary values. The methodology involves data preprocessing, model building using Support Vector Machines (SVC), Gaussian Naive Bayes, and Random Forest classifiers, model training and evaluation, disease prediction function development, pickling the files, loading pretrained models, setting Open API key, Flask connection, user's symptoms processing, connection with a database, and displaying disease information using OpenAI. The user receives complete information about the predicted disease through symptoms input, with SVC resulting in higher accuracy rates among the three models tested. The integration of machine learning and OpenAI provides a unique and comprehensive examination of symptoms, personalized to the user's needs.

# Dataset:
Contains 407 columns representing symptoms and 135 rows containing disease names.

Each column except the target column represents a symptom with binary values.

# Methodology:
Data preprocessing

Splitting and encoding

Model building: Support Vector Machines (SVC), Gaussian Naive Bayes, Random Forest

Model training and evaluation

Disease prediction function development

Pickling the files

Loading pretrained models

Setting Open API key

Flask connection

User's symptoms processing

Connection with a database

Function to get disease information using OpenAI

Displaying info to the user

# Python Libraries:

openai

numpy

pandas

matplotlib.pyplot

seaborn

LabelEncoder from sklearn.preprocessing

train_test_split from sklearn.model_selection

cross_val_score from sklearn.model_selection

sklearn

Flask 

sqlite3

pickle

array

importlib



# Results:
SVC resulted in higher accuracy rates among the three models tested.
Integration of machine learning and OpenAI provides a unique and comprehensive examination of symptoms, personalized to the user's needs.

# Conclusion:
The project demonstrates the potential of machine learning and OpenAI technologies in accurately predicting diseases based on symptoms, facilitating timely medical intervention and knowledge dissemination. By leveraging diverse machine learning algorithms and OpenAI capabilities, the system enhances the accuracy and reliability of disease predictions, providing a personalized and human-like experience to users.

# Future Work:
Integration of more advanced machine learning models and large language models.
Updating the data frequently and considering user feedback to improve model accuracy and predict a wider range of diseases efficiently.
