
# Importing libraries
import numpy as np
import pandas as pd
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.datasets import make_classification


# In[2]:


# Reading the train.csv by removing the 
# last column since it's an empty column
DATA_PATH = "main.csv"
data = pd.read_csv(DATA_PATH).dropna(axis = 1)

# Checking whether the dataset is balanced or not
disease_counts = data["label"].value_counts()
temp_df = pd.DataFrame({"Disease": disease_counts.index,"Counts": disease_counts.values})

plt.figure(figsize = (18,8))
sns.barplot(x = "Disease", y = "Counts", data = temp_df)
plt.xticks(rotation=90)
plt.show()


# In[ ]:





# In[3]:


# Encoding the target value into numerical
# value using LabelEncoder
encoder = LabelEncoder()
data["label"] = encoder.fit_transform(data["label"])


# In[4]:


data


# In[5]:


X = data.iloc[:,:-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size = 0.2, random_state = 24)

print(f"Train: {X_train.shape}, {y_train.shape}")
print(f"Test: {X_test.shape}, {y_test.shape}")


# In[6]:


from sklearn.model_selection import KFold, cross_val_score


# In[7]:


# Custom scoring function for cross-validation
def cv_scoring(estimator, X, y):
    predictions = estimator.fit(X, y).predict(X)
    print("Predictions:", predictions)  # Print predictions for debugging
    print("True Labels:", y.values)    # Print true labels for debugging
    return accuracy_score(y, predictions)

# Initialize Models
models = {
    "SVC": SVC(),
    "Gaussian NB": GaussianNB(),
    "Random Forest": RandomForestClassifier(random_state=18)
}

# Perform cross-validation for the models
for model_name in models:
    model = models[model_name]
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kfold, 
                             n_jobs=-1, 
                             scoring=cv_scoring)
    print("=" * 60)
    print(model_name)
    print(f"Scores: {scores}")
    print(f"Mean Score: {np.mean(scores)}")


# In[8]:


# Training and testing SVM Classifier
svm_model = SVC()
svm_model.fit(X_train, y_train)
preds = svm_model.predict(X_test)

print(f"Accuracy on train data by SVM Classifier\: {accuracy_score(y_train, svm_model.predict(X_train))*100}")
cf_matrix = confusion_matrix(y_test, preds)
plt.figure(figsize=(12,8))
sns.heatmap(cf_matrix, annot=True)
plt.title("Confusion Matrix for SVM Classifier on Test Data")
plt.show()


# In[9]:


nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
preds = nb_model.predict(X_test)
print(f"Accuracy on train data by Naive Bayes Classifier\: {accuracy_score(y_train, nb_model.predict(X_train))*100}")
cf_matrix = confusion_matrix(y_test, preds)
plt.figure(figsize=(12,8))
sns.heatmap(cf_matrix, annot=True)
plt.title("Confusion Matrix for Naive Bayes Classifier on Test Data")
plt.show()


# In[10]:


# Training and testing Random Forest Classifier
rf_model = RandomForestClassifier(random_state=18)
rf_model.fit(X_train, y_train)
preds = rf_model.predict(X_test)
print(f"Accuracy on train data by Random Forest Classifier: {accuracy_score(y_train, rf_model.predict(X_train))*100}")

cf_matrix = confusion_matrix(y_test, preds)
plt.figure(figsize=(12,8))
sns.heatmap(cf_matrix, annot=True)
plt.title("Confusion Matrix for Random Forest Classifier on Test Data")
plt.show()


# In[11]:


# Training the models on whole data
final_svm_model = SVC()
final_nb_model = GaussianNB()
final_rf_model = RandomForestClassifier(random_state=18)
final_svm_model.fit(X, y)
final_nb_model.fit(X, y)
final_rf_model.fit(X, y)


# In[ ]:





# In[12]:


# Reading the test data
test_data = pd.read_csv("main.csv")
 
test_X = test_data.iloc[:, :-1]
test_Y = encoder.transform(test_data.iloc[:, -1])
 
# Making prediction by take mode of predictions 
# made by all the classifiers
svm_preds = final_svm_model.predict(test_X)
nb_preds = final_nb_model.predict(test_X)
rf_preds = final_rf_model.predict(test_X)
from collections import Counter

def find_mode_prediction(predictions):
    counter = Counter(predictions)
    mode_prediction = counter.most_common(1)[0][0]
    return mode_prediction

final_preds = []
for i, j, k in zip(svm_preds, nb_preds, rf_preds):
    predictions = [i, j, k]
    mode_prediction = find_mode_prediction(predictions)
    final_preds.append(mode_prediction)
print(f"Accuracy on Test dataset by the combined model\: {accuracy_score(test_Y, final_preds)*100}")
sns.heatmap(cf_matrix, annot = True)
plt.title("Confusion Matrix for Combined Model on Test Dataset")
plt.show()


# In[13]:


symptoms = X.columns.values
symptom_index = {}
for index, value in enumerate(symptoms):
    symptom = " ".join([i.capitalize() for i in value.split("_")])
    symptom_index[symptom] = index

data_dict = {
    "symptom_index":symptom_index,
    "predictions_classes":encoder.classes_
}


# In[32]:


data_dict


# In[15]:


def predictDisease(symptoms):
    symptoms = symptoms.split(",")

    # creating input data for the models
    input_data = [0] * len(data_dict["symptom_index"])
    for symptom in symptoms:
        index = data_dict["symptom_index"].get(symptom)
        if index is not None:  # Check if symptom exists in the index
            input_data[index] = 1
        else:
            print(f"Symptom '{symptom}' not found in the index.")

    # reshaping the input data and converting it
    # into suitable format for model predictions
    input_data = np.array(input_data).reshape(1, -1)

    # generating individual outputs
    rf_prediction = data_dict["predictions_classes"][final_rf_model.predict(input_data)[0]]
    nb_prediction = data_dict["predictions_classes"][final_nb_model.predict(input_data)[0]]
    svm_prediction = data_dict["predictions_classes"][final_svm_model.predict(input_data)[0]]

    # Collect predictions from models
    predictions = [rf_prediction, nb_prediction, svm_prediction]

    # Finding the most frequent prediction
    final_prediction = max(set(predictions), key=predictions.count)

    result = {
        "rf_model_prediction": rf_prediction ,
        "naive_bayes_prediction": nb_prediction,
        "svm_model_prediction": svm_prediction,
        "final_prediction": final_prediction
    }
    
    print("RF Model Prediction:", rf_prediction)
    print("Naive Bayes Model Prediction:", nb_prediction)
    print("SVM Model Prediction:", svm_prediction)
    print("Final Prediction:", final_prediction)
    #return result

# Testing the modified function
print(predictDisease("Umls:c0010200 Cough,Umls:c0008033 Pleuritic pain,Umls:c0476273 Distress respiratory,Umls:c0239134 Productive cough,Umls:c0850149 Non-productive cough,Umls:c0436331 Symptom aggravating factors,Umls:c0232292 Chest tightness,Umls:c0043144 Wheezing,Umls:c0392680 Shortness of breath"))


# ### pickling

# In[16]:


import pickle


# In[17]:


# Save the Random Forest model
with open('final_rf_model.pkl', 'wb') as file:
    pickle.dump(final_rf_model, file)

# Save the Naive Bayes model
with open('final_nb_model.pkl', 'wb') as file:
    pickle.dump(final_nb_model, file)

# Save the SVM model
with open('final_svm_model.pkl', 'wb') as file:
    pickle.dump(final_svm_model, file)


# In[18]:


# Load the Random Forest model
pick_rf = pickle.load(open('final_rf_model.pkl', 'rb'))
pick_svm = pickle.load(open('final_svm_model.pkl', 'rb'))
pick_nb = pickle.load(open('final_nb_model.pkl', 'rb'))


# ## checking the pickle file

# In[19]:


symptoms1 = "Umls:c0010200 Cough,Umls:c0008033 Pleuritic pain,Umls:c0476273 Distress respiratory,Umls:c0239134 Productive cough,Umls:c0850149 Non-productive cough,Umls:c0436331 Symptom aggravating factors,Umls:c0232292 Chest tightness,Umls:c0043144 Wheezing,Umls:c0392680 Shortness of breath"

symptoms_list1 = symptoms1.split(",")
input_data1 = [0] * len(data_dict["symptom_index"])
for symptom in symptoms_list1:
    index1 = data_dict["symptom_index"].get(symptom)
    if index1 is not None:  # Check if symptom exists in the index
        input_data1[index1] = 1
    else:
        print(f"Symptom '{symptom}' not found in the index.")
input_data1 = np.array(input_data1).reshape(1, -1)


# In[20]:


print(input_data1)


# In[21]:


rf_predict = data_dict["predictions_classes"][pick_rf.predict(input_data1)[0]]
print("rf : "+rf_predict)
svm_predict = data_dict["predictions_classes"][pick_svm.predict(input_data1)[0]]
print("svm : "+svm_predict)
nb_predict = data_dict["predictions_classes"][pick_nb.predict(input_data1)[0]]
print("nb : "+nb_predict)


# In[22]:


import openai


# In[23]:


openai.api_key = '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' 


# In[24]:


def get_disease_info(disease_name,prompt):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=100
    )

    return response['choices'][0]['text'].strip()


# In[25]:


prompt_rf = f"Retrieve information about {rf_predict[14::]}."
prompt_nb = f"How {nb_predict[14::]} is spread and curavle methods, recomend any home medicines in points."
prompt_svm = f"Which doctor {svm_predict[14::]} effected people should meet and which age group effects the most."


# In[26]:


rf=get_disease_info(rf_predict[14::],prompt_rf)
nb=get_disease_info(nb_predict[14::],prompt_nb)
svm=get_disease_info(svm_predict[14::],prompt_svm)


# In[27]:


print(rf)
print(nb)
print(svm)


# In[ ]:




