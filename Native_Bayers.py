import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Titanic Survival Prediction", layout="wide")

st.title("🚢 Titanic Survival Prediction using Logistic Regression")

# -------------------------------
# Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Titanic-Dataset.csv")  # change path if needed
    return df

df = load_data()

# -------------------------------
# Show Raw Data
# -------------------------------
st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

# -------------------------------
# Data Information
# -------------------------------
with st.expander("📊 Dataset Information"):
    st.write(df.info())

# -------------------------------
# Missing Values
# -------------------------------
st.subheader("❓ Missing Values")
st.write(df.isnull().sum())

# -------------------------------
# Data Preprocessing
# -------------------------------
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

df.drop(['Cabin', 'Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)

df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# -------------------------------
# Feature & Target
# -------------------------------
X = df.drop('Survived', axis=1)
y = df['Survived']

# -------------------------------
# Train Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Model Training
# -------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -------------------------------
# Prediction
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# Model Evaluation
# -------------------------------
st.subheader("📈 Model Performance")

accuracy = accuracy_score(y_test, y_pred)
st.success(f"Accuracy: {accuracy:.2f}")

st.text("Classification Report")
st.text(classification_report(y_test, y_pred))

# -------------------------------
# Confusion Matrix
# -------------------------------
st.subheader("🔢 Confusion Matrix")

fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred),
            annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

st.pyplot(fig)

# -------------------------------
# Visualizations
# -------------------------------
st.subheader("📊 Data Visualizations")

col1, col2 = st.columns(2)

with col1:
    st.write("Survival Count")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=df, x='Survived', ax=ax1)
    st.pyplot(fig1)

with col2:
    st.write("Survival by Gender")
    fig2, ax2 = plt.subplots()
    sns.countplot(x='Sex_male', hue='Survived', data=df, ax=ax2)
    st.pyplot(fig2)

# -------------------------------
# User Prediction
# -------------------------------
st.subheader("🧍 Predict Survival (Custom Input)")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
age = st.slider("Age", 1, 80, 25)
sibsp = st.number_input("Siblings/Spouses", 0, 5, 0)
parch = st.number_input("Parents/Children", 0, 5, 0)
fare = st.slider("Fare", 0.0, 600.0, 50.0)
sex_male = st.selectbox("Gender", ["Female", "Male"])
embarked_q = st.selectbox("Embarked Q", [0, 1])
embarked_s = st.selectbox("Embarked S", [0, 1])

sex_male = 1 if sex_male == "Male" else 0

input_data = np.array([[pclass, age, sibsp, parch, fare,
                         sex_male, embarked_q, embarked_s]])

prediction = model.predict(input_data)

if st.button("Predict"):
    if prediction[0] == 1:
        st.success("🎉 Passenger Survived")
    else:
        st.error("❌ Passenger Did Not Survive")
