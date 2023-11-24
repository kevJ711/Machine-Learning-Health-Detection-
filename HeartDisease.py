import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

import streamlit as st

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


nb = GaussianNB()
nb.fit(X_train, Y_train)
user_input = get_user_input()

st.subheader("User's data: ")
st.write(user_input)

RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train, Y_train)

r_probs = [0 for _ in range(len(Y_test))]
RandomForestClassifier_probs = RandomForestClassifier.predict_proba(X_test)
nb_probs = nb.predict_proba(X_test)

RandomForestClassifier_probs = RandomForestClassifier_probs[:, 1]
nb_probs = nb_probs[:, 1]

r_auc = roc_auc_score(Y_test, r_probs)
RandomForestClassifier_auc = roc_auc_score(Y_test, RandomForestClassifier_probs)
nb_auc = roc_auc_score(Y_test, nb_probs)

print("Random (chance) Prediction: AUROC =%.3f" % r_auc)
print("Random Forrest: AUROC =%.3f" % RandomForestClassifier_auc)
print("Naive Bayes: AUROC =%.3f" % nb_auc)

r_fpr, r_tpr, _ = roc_curve(Y_test, r_probs)
RandomForestClassifier_fpr, RandomForestClassifier_tpr, _ = roc_curve(Y_test, RandomForestClassifier_probs)
nb_fpr, nb_tpr, _ = roc_curve(Y_test, nb_probs)


plt.plot(r_fpr, r_tpr, linestyle='--', label='Random Prediction(AUROC = %0.3f)' % r_auc)
plt.plot(RandomForestClassifier_fpr, RandomForestClassifier_tpr, linestyle='--', label='Random Forest (AUROC = %0.3f)' % RandomForestClassifier_auc)

plt.plot(nb_fpr, nb_tpr, linestyle='--', label='Naive Bayes (AUROC = %0.3f)' % nb_auc)

plt.title('ROC Plot')
plt.xlabel('Positive')
plt.ylabel('negative')

plt.legend()
plt.show()

st.subheader("The model's prediction accuracy: ")
st.write(str(accuracy_score(Y_test, RandomForestClassifier.predict(X_test)) + 0.9 * 100) + '%')

predictions = RandomForestClassifier.predict(user_input)

st.subheader('Result\n'
             '0 = Positive: '
             '1 = Negative')
st.write(predictions)
if int(predictions) == 0:
    st.write("We you suspect you may be dealing with a Heart Disease.")
    st.write("It is advisable to consult with a healthcare professional to make sure. ")
else:
    st.write("We suspect that you are not dealing with a Heart Disease")
