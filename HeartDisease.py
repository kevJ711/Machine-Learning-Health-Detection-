import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from PIL import Image
from xgboost import XGBClassifier  # Importing the XGBoost classifier
from sklearn.metrics import classification_report  # Classification report for model evaluation
from sklearn.model_selection import train_test_split  # Splitting data into training and testing sets
import streamlit as st
import plotly.graph_objects as go

from sklearn.metrics import roc_curve, auc


st.write('''
# Heart Disease Dectection Using Machine Learning 
Dectects if someone has a Heart Disease using the Public Health Dataset 
''')

image = Image.open('C:/Users/kevin/PycharmProjects/Python_Programs/HDlogo.PNG')
st.image(image, use_column_width=True)


df = pd.read_csv('C:/Users/kevin/PycharmProjects/Python_Programs/heart.csv')


st.subheader('Heart Disease Dataset')

st.dataframe(df)

st.write(df.describe())


chart = st.bar_chart(df)
st.scatter_chart(df)

X = df.iloc[:, 0:13].values
Y = df.iloc[:, -1].values
# split the data set into 75% training and 25% testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)


def get_user_input():
    age = st.sidebar.slider('age', 29, 77, 52)
    sex = st.sidebar.slider('sex: 0 = female, 1 = male', 0, 1, 0)
    cp = st.sidebar.slider('chest pain type: \n'
                           'Value 1 = typical angina, 2 = atypical angina,\n'
                           '3 = non-anginal pain, 4 = asymptomatic', 0, 3, 0)
    trestbps = st.sidebar.slider('resting blood pressure', 94, 200, 125)
    chol = st.sidebar.slider('serum cholestoral in mg/d', 126, 564, 212)
    fbs = st.sidebar.slider('fasting blood sugar > 120 mg/dl', 0, 1, 0)
    restecg = st.sidebar.slider('resting electrocardiographic results (values 0,1,2)', 0, 2, 1)
    thalach = st.sidebar.slider('maximum heart rate achieved', 71, 202, 161)
    exang = st.sidebar.slider('exercise induced angina', 0, 1, 0)
    oldpeak = st.sidebar.slider('oldpeak = ST depression induced by exercise relative to rest', 0.0, 6.2, 0.0)
    slope = st.sidebar.slider('the slope of the peak exercise ST segment', 0, 2, 2)
    ca = st.sidebar.slider('number of major vessels (0-3) colored by flourosopy', 0, 4, 1)
    thal = st.sidebar.slider('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect', 0, 3, 3)

    user_data = {'age': age,
                 'sex': sex,
                 'chest pain type': cp,
                 'resting blood pressure': trestbps,
                 'chol': chol,
                 'fbs': fbs,
                 'restecg': restecg,
                 'thalach': thalach,
                 'exang': exang,
                 'oldpeak': oldpeak,
                 'slope': slope,
                 'ca': ca,
                 'thal': thal

                 }

    features = pd.DataFrame(user_data, index=[0])
    return features


user_input = get_user_input()

st.subheader("User's data: ")
st.write(user_input)

RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train, Y_train)


naive_bayes_classifier = GaussianNB()
naive_bayes_classifier.fit(X_train, Y_train)

xgb_classifier = XGBClassifier()
xgb_classifier.fit(X_train, Y_train)


st.subheader("The model's prediction accuracy: ")
st.write(str(accuracy_score(Y_test, RandomForestClassifier.predict(X_test)) * 100) + '%')

# Evaluate Random Forest Classifier
st.subheader("Random Forest Classifier:")
rf_accuracy = accuracy_score(Y_test, RandomForestClassifier.predict(X_test))
st.write(f"Accuracy: {rf_accuracy * 100:.2f}%")
st.write("Classification Report:")
st.write(classification_report(Y_test, RandomForestClassifier.predict(X_test)))

# Evaluate Naive Bayes Classifier
st.subheader("Naive Bayes Classifier:")
nb_accuracy = accuracy_score(Y_test, naive_bayes_classifier.predict(X_test))
st.write(f"Accuracy: {nb_accuracy * 100:.2f}%")
st.write("Classification Report:")
st.write(classification_report(Y_test, naive_bayes_classifier.predict(X_test)))



# Evaluate XGBooster Classifier

st.subheader("XGBooster Classifier:")
xgb_accuracy = accuracy_score(Y_test, xgb_classifier.predict(X_test))
st.write(f"Accuracy: {xgb_accuracy * 100:.2f}%")
st.write("Classification Report:")
st.write(classification_report(Y_test, xgb_classifier.predict(X_test)))


# Compare all the classifiers

st.subheader("Comparison:")
st.write(f"Random Forest Classifier Accuracy: {rf_accuracy + 0.9 * 100:.2f}%")
st.write(f"Naive Bayes Classifier Accuracy: {nb_accuracy * 100:.2f}%")
st.write(f"XGBooster Classifier Accuracy: {xgb_accuracy * 100:.2f}%")

# Getting the predicted probabilities for all the classifiers:
rf_probs = RandomForestClassifier.predict_proba(X_test)[:, 1]
nb_probs = naive_bayes_classifier.predict_proba(X_test)[:, 1]
xgboost_probs = xgb_classifier.predict_proba(X_test)[:, 1]
# Computing the Receiver Operating Characteristic (ROC) curve for the Random Forest model
rf_fpr, rf_tpr, _ = roc_curve(Y_test, rf_probs)
# Compute the Area Under the Curve (AUC) for the ROC curve of the Random Forest model
rf_auc = auc(rf_fpr, rf_tpr)
# Compute the Receiver Operating Characteristic (ROC) curve for the Naive Bayes model
nb_fpr, nb_tpr, _ = roc_curve(Y_test, nb_probs)
# Compute the Area Under the Curve (AUC) for the ROC curve of the Naive Bayes model
nb_auc = auc(nb_fpr, nb_tpr)
# Compute the Receiver Operating Characteristic (ROC) curve for the XGbooster model

xgboost_fpr, xgboost_tpr, _ = roc_curve(Y_test, xgboost_probs)
# Compute the Area Under the Curve (AUC) for the ROC curve of the XGbooster model

xgboost_auc = auc(xgboost_fpr, xgboost_tpr)
# Create a Plotly figure to display
fig = go.Figure()

# Adding the Random Forest ROC curve
fig.add_trace(go.Scatter(x=rf_fpr, y=rf_tpr, mode='lines', name=f'Random Forest (AUC={rf_auc:.2f})'))

# Adding the  Naive Bayes ROC curve
fig.add_trace(go.Scatter(x=nb_fpr, y=nb_tpr, mode='lines', name=f'Naive Bayes (AUC={nb_auc:.2f})'))
# Adding the XGBoost ROC curve
fig.add_trace(go.Scatter(x=xgboost_fpr, y=xgboost_tpr, mode='lines', name=f'XGBoost (AUC={xgboost_auc:.2f})'))

# Seting the the layout
fig.update_layout(
    title='Receiver Operating Characteristic (ROC) Curve',
    xaxis=dict(title='False Positive Rate'),
    yaxis=dict(title='True Positive Rate'),
    legend=dict(x=0.7, y=0.2),
)
# Create a Plotly figure for ROC curve
fig = go.Figure()

st.plotly_chart(fig)

# Adding the Random Forest ROC curve
fig.add_trace(go.Scatter(x=rf_fpr, y=rf_tpr, mode='lines', name=f'Random Forest (AUC={rf_auc:.2f})'))

# Adding the  Naive Bayes ROC curve
fig.add_trace(go.Scatter(x=nb_fpr, y=nb_tpr, mode='lines', name=f'Naive Bayes (AUC={nb_auc:.2f})'))
# Adding the XGBoost ROC curve
fig.add_trace(go.Scatter(x=xgboost_fpr, y=xgboost_tpr, mode='lines', name=f'XGBoost (AUC={xgboost_auc:.2f})'))

# Seting the the layout
fig.update_layout(
    title='Receiver Operating Characteristic (ROC) Curve',
    xaxis=dict(title='False Positive Rate'),
    yaxis=dict(title='True Positive Rate'),
    legend=dict(x=0.7, y=0.2),
)
# Display the results of the random forest model
print('Result for Random Forest\n0 = Positive, 1 = Negative')
print(f"Accuracy for Random Forest Classifier: {rf_accuracy * 100:.2f}%")

if int(rf_accuracy) == 0:
    print("By using the rf model We suspect you may be dealing with a Heart Disease.")
    print("It is advisable to consult with a healthcare professional to make sure.")
else:
    print("By using the rf model We suspect that you are not dealing with a Heart Disease")

# Displaying the predicted result for Naive Bayes
st.subheader('\nResult for Naive Bayes\n0 = Positive, 1 = Negative')
st.subheader(f"Accuracy for Naive Bayes Classifier: {nb_accuracy * 100:.2f}%")

if int(nb_accuracy) == 0:
    st.subheader("By using the Naive Bayes model We suspect you may be dealing with a Heart Disease.")
    st.subheader("It is advisable to consult with a healthcare professional to make sure.")
else:
    st.subheader("By using the Naive Bayes We suspect that you are not dealing with a Heart Disease")
st.subheader('Result\n'
             '0 = Positive: '
             '1 = Negative')
