# Machine-Learning-Health-Detection-
Dectects if someone has a Heart Disease with python using the Public Health Dataset of heart diseases.

My Machine Learning Model that predicts if a user would be diagnosed with heart diseases based on the Public Health Dataset via https://www.kaggle.com/


Primary goal: 
The primary goal is to develop a predictive model to determine the likelihood of a user being diagnosed with heart disease based on relevant health-related features.

Dataset Source: 
The model is trained on a public health dataset sourced from heart.csv. This dataset comprises hundreds of instances related to individuals' health. This data set dates from 1988 and consists of four databases: Cleveland, Hungary, Switzerland, and Long Beach V. 

The dataset includes features such as age
sex
chest pain type (4 values)
resting blood pressure
serum cholestoral in mg/dl
fasting blood sugar > 120 mg/dl
resting electrocardiographic results (values 0,1,2)
maximum heart rate achieved
exercise induced angina
oldpeak = ST depression induced by exercise relative to rest
the slope of the peak exercise ST segment
number of major vessels (0-3) colored by flourosopy
thal: 0 = normal; 1 = fixed defect; 2 = reversable defect The names and social security numbers of the patients were recently removed from the database, replaced with dummy values.

The target variable is the diagnosis of heart disease, represented as 0 or 1, where 0 indicates no heart disease and 1 indicates the presence of heart disease.


Model Selection and Training: 
The Random Forest Classifier algorithm was selected for its suitability in binary classification tasks.

The dataset was split into training and testing sets (80/20 split), and the model was trained on the training set.

This model can be deployed in a web application using scikit_learn 

IF you have any question, comments, or concerns please reach out! 
  
