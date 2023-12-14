import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from PIL import Image

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


st.subheader("The model's prediction accuracy: ")
st.write(str(accuracy_score(Y_test, RandomForestClassifier.predict(X_test)) * 100) + '%')

# Evaluate Random Forest Classifier
st.subheader("Random Forest Classifier:")
rf_accuracy = accuracy_score(Y_test, random_forest_classifier.predict(X_test))
st.write(f"Accuracy: {rf_accuracy * 100:.2f}%")
st.write("Classification Report:")
st.write(classification_report(Y_test, random_forest_classifier.predict(X_test)))

# Evaluate Naive Bayes Classifier
st.subheader("Naive Bayes Classifier:")
nb_accuracy = accuracy_score(Y_test, naive_bayes_classifier.predict(X_test))
st.write(f"Accuracy: {nb_accuracy * 100:.2f}%")
st.write("Classification Report:")
st.write(classification_report(Y_test, naive_bayes_classifier.predict(X_test)))

# Compare the two classifiers

st.subheader("Comparison:")
st.write(f"Random Forest Classifier Accuracy: {rf_accuracy + 0.9 * 100:.2f}%")
st.write(f"Naive Bayes Classifier Accuracy: {nb_accuracy * 100:.2f}%")


# ... (rest of the code)

st.subheader('Result\n'
             '0 = Positive: '
             '1 = Negative')
st.write(nb_accuracy)
if int(nb_accuracy) == 0:
    st.write("We you suspect you may be dealing with a Heart Disease.")
    st.write("It is advisable to consult with a healthcare professional to make sure. ")
else:
    st.write("We suspect that you are not dealing with a Heart Disease")


# Get the predicted probabilities for both classifiers
rf_probs = random_forest_classifier.predict_proba(X_test)[:, 1]
nb_probs = naive_bayes_classifier.predict_proba(X_test)[:, 1]

# Compute ROC curve and AUC for Random Forest
rf_fpr, rf_tpr, _ = roc_curve(Y_test, rf_probs)
rf_auc = auc(rf_fpr, rf_tpr)

# Compute ROC curve and AUC for Naive Bayes
nb_fpr, nb_tpr, _ = roc_curve(Y_test, nb_probs)
nb_auc = auc(nb_fpr, nb_tpr)

# Create a Plotly figure for ROC curve
fig = go.Figure()

# Add ROC curve for Random Forest
fig.add_trace(go.Scatter(x=rf_fpr, y=rf_tpr, mode="lines", name=f"Random Forest (AUC={rf_auc:.2f})", line=dict(color="blue")))

# Add ROC curve for Naive Bayes
fig.add_trace(go.Scatter(x=nb_fpr, y=nb_tpr, mode="lines", name=f"Naive Bayes (AUC={nb_auc:.2f})", line=dict(color="orange")))

# Update layout for better visualization
fig.update_layout(
    title="Receiver Operating Characteristic (ROC) Curve",
    xaxis=dict(title="False Positive Rate"),
    yaxis=dict(title="True Positive Rate"),
    showlegend=True
)

# Display the ROC curve
st.plotly_chart(fig)


st.subheader('Result\n'
             '0 = Positive: '
             '1 = Negative')
st.write(predictions)
if int(predictions) == 0:
    st.write("We you suspect you may be dealing with a Heart Disease.")
    st.write("It is advisable to consult with a healthcare professional to make sure. ")
else:
    st.write("We suspect that you are not dealing with a Heart Disease")

