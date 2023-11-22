
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
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


user_input = get_user_input()

st.subheader("User's data: ")
st.write(user_input)

RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train, Y_train)


st.subheader("The model's prediction accuracy: ")
st.write(str(accuracy_score(Y_test, RandomForestClassifier.predict(X_test)) * 100) + '%')


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

