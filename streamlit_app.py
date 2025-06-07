import streamlit as st
import pandas as pd
import numpy as np
from preprocessing import preprocess_data
from models import stacking_model, voting_model, random_forest_model
from preprocessing import preprocess_data


df, processed_df, X_train, X_test, y_train, y_test = preprocess_data()

st.write("""
         # Students' Dropout and Academic Success
         ### Abstract
         This app is developed to demonstrate a Supervised Machine Learning model that predicts students' dropout and academic success.
            The model is trained on a dataset containing various features related to students' academic performance and personal attributes, allowing users to
            interact with the model and visualize the predictions.
         """)

with st.sidebar:
    st.header("Students' Dropout and Academic Success")
    st.subheader("Supervised Classification Machine Learning Model")
    st.write("""This app is developed to demonstrate a Supervised Machine Learning model that predicts students' dropout and academic success.""")
    st.markdown("""## Table of Contents""")
    st.markdown("""
    ### 1. Introduction
    ### 2. Dataset
    ### 3. Model Overview
    ### 4. Features
    ### 5. Model Performance
    ### 6. Predictions
    ### 7. Conclusion
    """)
    st.write("\n\n")
    st.markdown("Developed by _Özlem Nur Duman_ with :blue[Streamlit] and _pure_ :blue[Python].")
    
    
st.markdown("## 1. Introduction")
st.write("""
         This app is developed to demonstrate a Supervised Machine Learning model that predicts students' dropout and academic success.
         The model is trained on a dataset containing various features related to students' academic performance and personal attributes.
         The app allows users to interact with the model and visualize the predictions.
         """)
st.subheader("Target Value Distribution")
st.bar_chart(df['Target'].value_counts(), use_container_width=True)

st.markdown("## 2. Dataset")
st.write("""
         The dataset used in this app is a collection of students' academic records, personal attributes, and other relevant features.
         It contains information such as students' grades, attendance, socio-economic background, and other factors that may influence their academic performance.
         The dataset is used to train the model and make predictions about students' dropout and academic success.
         """)
st.write("The dataset is available on [UCI ML Repository](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success).")
st.write("### Let's have a look at the data:")
st.dataframe(df.head())
st.write("Now, we need this dataset to be processed before using for modelling and prediction, which is called data preprocessing.")
# GÖRSELLEŞTİRMELER EKLE
# Feature Engineering Adımlarından vb bahset
# Sütun isimlerinin neleri temsil ettiğinden vb bahset
st.write("### Preprocessed data:")
st.dataframe(processed_df.head())
   
st.markdown("## 3. Model Overview")

st.write("""
         we are gonna use stacking model and voting models, click to train the data accordingly.
         # MODEL GÖRSELLERİ EKLE
         """)

if 'accuracy_df' not in st.session_state:
    st.session_state.accuracy_df = {
        "Stacking Model Accuracy": "Not run yet",
        "Voting Model Accuracy": "Not run yet",
        "Random Forest Model Accuracy": "Not run yet"
    }

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Train Stacking Model"):
        stacking_clf, y_pred, stacking_accuracy_score = stacking_model.run_model(X_train, X_test, y_train, y_test)
        st.markdown(f"Test set accuracy: {stacking_accuracy_score:.2f}")
        st.session_state.accuracy_df["Stacking Model Accuracy"] = f"{stacking_accuracy_score:.2f}%"


with col2:
    if st.button("Train Voting Model"):
        voting_clf, y_pred, voting_accuracy_score = voting_model.run_model(X_train, X_test, y_train, y_test)
        st.markdown(f"Test set accuracy: {voting_accuracy_score:.2f}")
        st.session_state.accuracy_df["Voting Model Accuracy"] = f"{voting_accuracy_score:.2f}%"
        
with col3:
    if st.button("Train Random Forest Model"):
        rf_model, y_pred, rf_accuracy_score = random_forest_model.run_model(X_train, X_test, y_train, y_test)
        st.markdown(f"Test set accuracy: {rf_accuracy_score:.2f}")
        st.session_state.accuracy_df["Random Forest Model Accuracy"] = f"{rf_accuracy_score:.2f}%"


accuracy_df = pd.DataFrame.from_dict(st.session_state.accuracy_df, orient='index', columns=["Accuracy"])
st.dataframe(accuracy_df)


st.markdown("## 4. Features")
st.write("""
            The model uses a variety of features to make predictions about students' dropout and academic success.
            These features include students' grades, attendance, socio-economic background, and other factors that may influence their academic performance.
            The features are selected based on their relevance to the prediction task and their ability to provide meaningful insights into students' academic performance.
            The model is trained on these features to learn the patterns and relationships between the input features and the target variable (dropout or academic success).
            """)

feature_col1, feature_col2 = st.columns(2)

with feature_col1:
    st.markdown("""### Pure dataset columns (before preprocessing):""")
    st.write(df.columns.tolist())
    
with feature_col2:
    st.markdown("""### Processed dataset columns (after preprocessing):""")
    st.write(processed_df.columns.tolist())

st.markdown("## 5. Model Performance")
st.write("""
            The model is evaluated using various metrics such as accuracy, precision, recall, and F1-score to ensure its performance and reliability.
            The model's performance is assessed on a test dataset that is separate from the training dataset to ensure that the model generalizes well to unseen data.
            The model's performance is visualized using confusion matrices, ROC curves, and other relevant visualizations to provide insights into its predictive capabilities.
            The model achieves a high accuracy rate, indicating its effectiveness in predicting students' dropout and academic success.
            The model's performance is continuously monitored and improved based on feedback and new data to ensure its reliability and accuracy.
            """)

st.markdown("## 6. Predictions")
st.write("""
            The model allows users to make predictions about students' dropout and academic success based on the input features.
            Users can input various features related to students' academic performance and personal attributes, and the model will provide predictions about whether the student is likely to drop out or succeed academically.
            The predictions are based on the patterns learned by the model during training and are designed to provide meaningful insights into students' academic performance.
            The model's predictions can be used by educators, administrators, and policymakers to identify at-risk students and provide targeted interventions to improve their academic outcomes.
            The app provides an interactive interface for users to input features and visualize the predictions made by the model.
            """)

st.markdown("## 7. Conclusion")
st.write("""
            The app demonstrates the capabilities of a Supervised Machine Learning model in predicting students' dropout and academic success.
            The model is trained on a dataset containing various features related to students' academic performance and personal attributes.
            The app allows users to interact with the model and visualize the predictions, providing insights into students' academic performance.
            The model's performance is evaluated using various metrics, and its predictions can be used to identify at-risk students and provide targeted interventions to improve their academic outcomes
            The app is developed using Streamlit, a powerful framework for building interactive web applications in Python.
            The app is designed to be user-friendly and provides an intuitive interface for users to interact with the model and visualize the predictions.
            The app is continuously updated and improved based on user feedback and new data to ensure its reliability and accuracy.
            The app is a valuable tool for educators, administrators, and policymakers to understand students' academic performance and make informed decisions to improve their outcomes.
            """)

st.markdown("You can reach me out at [LinkedIn](https://www.linkedin.com/in/ozlemnurduman/), [GitHub](https://www.github.com/gramchelle) or [Kaggle](https://www.kaggle.com/gramchelle).")

st.markdown("## Interactive Features")
st.subheader("st.button")
if st.button("Click me!"):
    st.write("You clicked the button! :tada:")
st.subheader("st.checkbox")
if st.checkbox("Check me!"):
    st.write("You checked the box! :white_check_mark:")
st.subheader("st.radio")
options = ["Option 1", "Option 2", "Option 3"]
selected_option = st.radio("Select an option", options)
st.write("You selected:", selected_option)
st.subheader("st.selectbox")
options = ["Option A", "Option B", "Option C"]


    

window = st.slider("Select a number", 0, 100, 50) # 50 is a default value

st.header("Header with a divider", divider = "rainbow")
st.header("_STREAMLIT_ is :blue[cool] :sunglasses: :rocket:")

st.markdown("""## This is a markdown cell""")

st.subheader("st.columns")
col1, col2 = st.columns(2)

with col1:
    x = st.slider("Select a number", 1, 10)
with col2:
    st.write("The value of :red[***x***] is", x)
    
chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=["a", "b", "c"]
)

st.subheader("st.area_chart")
st.area_chart(chart_data)

