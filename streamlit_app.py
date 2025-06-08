import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from preprocessing import preprocess_data, smote_data, feature_importances
from models import lightgbm_voting_model, stacking_model, voting_model, random_forest_model
from sklearn.metrics import confusion_matrix, classification_report


df, processed_df, X_train, X_test, y_train, y_test = preprocess_data()
df_copy = df.copy()

st.write("""
         # Students' Dropout and Academic Success :rocket:
         ### Abstract
         This application demonstrates a supervised machine learning model designed to predict student dropout and academic performance. 
         Trained on a dataset comprising diverse academic and personal attributes of students, the model enables users to interactively 
         explore predictions and visualize outcomes through an intuitive interface.
         """)

st.sidebar.title("🎓 Student Dropout & Success Predictor")
st.sidebar.markdown("*A Supervised Classification ML App*")

st.sidebar.markdown("---")

st.sidebar.subheader("📊 Purpose")
st.sidebar.markdown(
    "An interactive app built with **Streamlit** and **Python**, showcasing a machine learning model that predicts "
    "**student dropout** and **academic success** based on key features."
)

##### SIDEBAR #####
st.sidebar.markdown("---")

st.sidebar.subheader("📚 Navigate the App")
st.sidebar.markdown("""
### 📌 Table of Contents
1. **Introduction**  
   Understand the problem and the goal of this project.

2. **Dataset**  
   Learn about the data source and its key attributes.

3. **Data Preprocessing & Feature Engineering**  
   See how raw data is cleaned and transformed for modeling.

4. **Features**  
   Explore the most relevant variables used in prediction.

5. **Model Overview**  
   Discover the machine learning models applied.

6. **Model Performance**  
   Analyze metrics and visualizations to assess effectiveness.

7. **Predictions**  
   Interact with the model and see prediction results.
   
8. **Conclusion**  
   Summarize findings, limitations, and future improvements.
   
9. References

10. About the Developer
""")

st.sidebar.markdown("---")

st.sidebar.markdown("👩‍💻 *Developed with care by*  \n**Özlem Nur Duman**")
st.sidebar.markdown("💡 Powered by **Streamlit** & **Python**")

st.sidebar.markdown("---")

st.sidebar.markdown("### 🎯 *\"Empowering education through intelligent prediction.\"*")
    
##### HEADER: Introduction #####
st.markdown("## 1. Introduction")
st.write("""
        This web application showcases the implementation of a supervised machine learning model developed to predict student dropout likelihood 
        and academic success. By leveraging a dataset enriched with features related to students' educational history, demographic background, 
        and performance metrics, the model offers data-driven insights into academic outcomes. Users can interact with the trained model, examine 
        prediction results, and gain a better understanding of the factors influencing student retention and achievement.
         """)
st.subheader("Target Value Distribution")
st.bar_chart(df['Target'].value_counts(), use_container_width=True)

##### HEADER: Dataset #####
st.markdown("## 2. Dataset")
st.write("""
         The dataset used in this app is a collection of students' academic records, personal attributes, and other relevant features.
         It contains information such as students' grades, attendance, socio-economic background, and other factors that may influence their academic performance.
         The dataset is used to train the model and make predictions about students' dropout and academic success.
         
         This dataset contains:
        * **Number of rows**: {len(df)}  
        * **Number of columns**: {len(df.columns)}  
        * **Target variable**: 'Target' (Dropout, Enrolled, Graduate)""")

st.write("The dataset is available on [UCI ML Repository - Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success).")
st.write("### Let's have a look at the data:")
st.dataframe(df.head())
st.markdown("Most of the columns are self-explanatory, but some of them are not. Let's have a look at the columns and their meanings:")
st.markdown("""
### 📋 Dataset Column Descriptions
**0. Marital status**: Student's marital status  
**1. Application mode**: Method of application (e.g., online, ordinance, transfer, etc.)  
**2. Application order**: Priority order in the application  
**3. Course**: Code of the course student enrolled in  
**4. Daytime/evening attendance**: Whether the student attends during the day or evening  
**5. Previous qualification**: Type of previous academic qualification  
**6. Previous qualification (grade)**: Grade obtained in previous qualification  
**7. Nacionality**: Student's nationality  
**8. Mother's qualification**: Mother's education level  
**9. Father's qualification**: Father's education level  
**10. Mother's occupation**: Mother's job category  
**11. Father's occupation**: Father's job category  
**12. Admission grade**: Grade at the time of admission  
**13. Displaced**: Whether the student lives away from their usual home location  
**14. Educational special needs**: Whether the student has special educational needs  
**15. Debtor**: Indicates if the student has outstanding payments  
**16. Tuition fees up to date**: Whether tuition fees are paid on time  
**17. Gender**: Gender of the student  
**18. Scholarship holder**: Whether the student holds a scholarship  
**19. Age at enrollment**: Student's age at the time of enrollment  
**20. International**: Whether the student is an international student  
**21. Curricular units 1st sem (credited)**: Number of credited courses in the 1st semester  
**22. Curricular units 1st sem (enrolled)**: Number of enrolled courses in the 1st semester  
**23. Curricular units 1st sem (evaluations)**: Number of course evaluations attended in the 1st semester  
**24. Curricular units 1st sem (approved)**: Number of courses passed in the 1st semester  
**25. Curricular units 1st sem (grade)**: Average grade in the 1st semester  
**26. Curricular units 1st sem (without evaluations)**: Courses without evaluation in the 1st semester  
**27. Curricular units 2nd sem (credited)**: Number of credited courses in the 2nd semester  
**28. Curricular units 2nd sem (enrolled)**: Number of enrolled courses in the 2nd semester  
**29. Curricular units 2nd sem (evaluations)**: Number of course evaluations attended in the 2nd semester  
**30. Curricular units 2nd sem (approved)**: Number of courses passed in the 2nd semester  
**31. Curricular units 2nd sem (grade)**: Average grade in the 2nd semester  
**32. Curricular units 2nd sem (without evaluations)**: Courses without evaluation in the 2nd semester  
**33. Unemployment rate**: National unemployment rate at the time of enrollment  
**34. Inflation rate**: National inflation rate at the time of enrollment  
**35. GDP**: Gross Domestic Product indicator at the time of enrollment  
**36. Target**: Final status of the student – graduate, dropout, or still enrolled  
""")
st.write("Before using this dataset for modeling and prediction, it must first undergo data preprocessing to ensure it is clean, consistent, and ready for analysis.")
st.write("### Preprocessed data:")
st.dataframe(processed_df.head())
# st.header("", divider="gray")

###### HEADER: Data Preprocessing and Feature Engineering #####
st.markdown("## 3. Data Preprocessing and Feature Engineering")

st.markdown("""
1. **Read Raw CSV**  
- Load the original CSV file (semicolon-separated) into a Pandas DataFrame called `df`.  
- Keep a copy named `non_processed_df` for reference.

2. **Encode Target Variable**  
   - Use `LabelEncoder` to transform the `"Target"` column (e.g., “dropout”/“graduate”/“enrolled”) into integer labels.

3. **Application Order Adjustment**  
   - Replace any `0 → 1` and `9 → 5` in the `"Application order"` column to collapse rarely-used codes into a standard range.

4. **Age Binning**  
   - Map `"Age at enrollment"` (actual years) into discrete buckets (0–11) based on these intervals:  
     - 17–18 → 0,  19 → 1,  20 → 2,  21–23 → 3,  24–27 → 4,  28–30 → 5,  31–33 → 6,  34–36 → 7,  37–41 → 8,  42–45 → 9,  46–50 → 10,  51+ → 11.

5. **Marital Status Cleanup**  
   - Map any code `6 → 4` and `3 → 4` in `"Marital status"` so that rare categories collapse into a single “Other” category.

6. **Application Mode Consolidation**  
   - Replace codes `{2, 5, 26, 27, 57} → 10` in `"Application mode"`.  
     This collapses various special-contingent, transfer, or international subcodes into one “Other” bucket (value 10).

7. **Course Code Correction**  
   - Replace any `33 → 9556` in `"Course"` to correct a mislabeled program code.

8. **Previous Qualification Simplification**  
   - Define a helper `simplify_qualification(x)` that maps raw numeric codes into four buckets:  
     - `1 → "Secondary"`  
     - `[2,3,4,5,6,39,40,42,43] → "Higher"`  
     - `[9,10,12,14,15,19,38] → "Basic"`  
     - Otherwise → "Other".  
   - Convert that bucket string into integers `{ "Basic": 0, "Secondary": 1, "Higher": 2, "Other": 3 }`.

9. **Compute Combined Previous Qualification Score**  
   - Create `"previous_qualification"` = (`"Previous qualification (grade)"` / 200) × (`simplified_previous_qualification` integer).  
   - Then drop the original `"Previous qualification (grade)"` and the intermediate `"Previous qualification"` buckets.

10. **Nationality Binarization**  
    - Convert `"Nacionality"` = 1 if code == 1 (local), else 0 (international).  
    - Drop the original `"Nacionality"` column since it’s now binary.

11. **Mother’s & Father’s Qualification Grouping**  
    - Define `group_mother_qualification(x)` and `group_father_qualification(x)` functions to map each parent’s raw code into one of 5 ordinal levels {0…4}.  
    - Convert those group labels to `int`.

12. **Mother’s & Father’s Occupation Grouping**  
    - Define `group_mother_occupation(x)` and `group_father_occupation(x)` to map each parent’s raw occupation codes into categories such as:  
      - “Unskilled”, “Skilled_Manual”, “Technical”, “Services_Sales”, “Administrative”, “Professional”, “Management”, “Armed_Forces” (father only), “Operator_Driver” (father only), “Other_or_Student”, or “Other.”  
    - Apply `LabelEncoder` to transform those string categories into integer codes.

13. **Drop Unused Columns**  
    - Remove `"International"` (if present).  
    - Drop the six “Curricular units 1st sem” and six “Curricular units 2nd sem” raw columns after computing derived metrics (see next step).  
    - Drop the raw `"Mother's occupation"`, `"Mother's qualification"`, `"Father's occupation"`, and `"Father's qualification"` after computing combined “knowledge” features (see below).

14. **Inflation Rate & GDP Categorization**  
    - Map `"Inflation rate"` numeric values into integers `{ -0.8→0,  -0.3→1,  0.3→2,  0.5→3,  0.6→4,  1.4→5,  2.6→6,  2.8→7,  3.7→8 }`.  
    - Map `"GDP"` values into `{ -4.06→0,  -3.12→1,  -1.70→2,  -0.92→3,  0.32→4,  0.79→5,  1.74→6,  1.79→7,  2.02→8,  3.51→9 }`.

15. **Admission Grade Outlier Removal (IQR Clipping)**  
    - Compute Q1, Q3 for `"Admission grade"`.  
    - Remove any rows where `"Admission grade"` is outside `[Q1 – 1.5×IQR,  Q3 + 1.5×IQR]`.

16. **Derived Semester‐Based Features**  
    - `sem_1_pass_rate` = `("Curricular units 1st sem (approved)" ÷ "Curricular units 1st sem (enrolled)")`  
    - `sem_2_pass_rate` = `("Curricular units 2nd sem (approved)" ÷ "Curricular units 2nd sem (enrolled)")`  
    - `sem_1_points_per_credit` = `("Curricular units 1st sem (grade)" ÷ "Curricular units 1st sem (approved)")`  
    - `sem_2_points_per_credit` = `("Curricular units 2nd sem (grade)" ÷ "Curricular units 2nd sem (approved)")`  
    - `sem1_success_rate` = `("Curricular units 1st sem (credited)" ÷ "Curricular units 1st sem (enrolled)")`  
    - `sem2_success_rate` = `("Curricular units 2nd sem (credited)" ÷ "Curricular units 2nd sem (enrolled)")`  
    - `sem1_evaluation_rate` = `(("Curricular units 1st sem (enrolled)" – "Curricular units 1st sem (without evaluations)") ÷ "Curricular units 1st sem (enrolled)")`  
    - `sem2_evaluation_rate` = `(("Curricular units 2nd sem (enrolled)" – "Curricular units 2nd sem (without evaluations)") ÷ "Curricular units 2nd sem (enrolled)")`  
    - `avg_evaluations` = `( "Curricular units 1st sem (evaluations)" + "Curricular units 2nd sem (evaluations)" ) / 2`

17. **Parent Knowledge Features**  
    - `father_knowledge` = `("Father's occupation code" × "Father's qualification code") / 2`  
    - `mother_knowledge` = `("Mother's occupation code" × "Mother's qualification code") / 2`

18. **Train/Test Split**  
    - Define `label = "Target"` and drop it from the feature set.  
    - Split into `X_train, X_test, y_train, y_test` using `train_test_split(stratify=y, test_size=0.2, random_state=42)`.

19. **SMOTE (Optional, Later Step)**  
    - After this function returns, SMOTE can be applied to `X_train, y_train` if `st.session_state.smote_enabled` is set.

After completing these steps, the returned `processed_df` (and its train/test subsets) contain only clean, transformed features ready for modeling.
""")

def visualization(col):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    sns.histplot(df[col], kde=True, ax=axes[0, 0])
    axes[0, 0].set_title(f'Distribution of {col}')

    sns.boxplot(x=df["Target"], y=df[col], ax=axes[0, 1])
    axes[0, 1].set_title(f'Boxplot of {col} by Target')

    sns.histplot(processed_df[col], kde=True, ax=axes[1, 0])
    axes[1, 0].set_title(f'Distribution of {col} after IQR Clipping')

    sns.boxplot(x=processed_df["Target"], y=processed_df[col], ax=axes[1, 1])
    axes[1, 1].set_title(f'IQR Clipped {col} by Target')

    plt.tight_layout()
    st.pyplot(fig)
    
visualization("Admission grade")

def two_histplots(df, col1, col2):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5)) #1d
    
    sns.histplot(df[col1], kde=True, ax=axes[0])
    axes[0].set_title(f'Distribution of {col1}')
    
    sns.histplot(df[col2], kde=True, ax=axes[1])
    axes[1].set_title(f'Distribution of {col2}')
    
    st.pyplot(fig)


two_histplots(processed_df, "sem_1_pass_rate", "sem_2_pass_rate")

#####  HEADER: Model Overview #####
st.markdown(""" ## 4. Model Overview""")

import streamlit as st

st.markdown("""

In this project, we experimented with several ensemble learning strategies to improve our multi-class classification performance. Below is an overview of the models used, their architecture, and the rationale behind them.

---

### :rocket: 1. Stacking Classifier (Best Performing Model)

We used a **Stacking Classifier** that combines predictions from multiple base models (XGBoost and Random Forest) using a meta-model (Logistic Regression). This model outperformed others in terms of accuracy.

**Why Stacking?**  
Stacking helps to reduce model bias and variance by combining the strengths of different algorithms. The base models capture diverse patterns, and the meta-learner synthesizes this information to make better generalizations.

**Model Components:**
- **XGBoost:** Captures non-linear relationships efficiently.
- **Random Forest:** Reduces overfitting through bootstrapped aggregation.
- **Logistic Regression:** Combines base predictions in a linear, interpretable way.

---

### :rocket: 2. Voting Classifier (Alternative Ensemble Method)

Another ensemble technique used was **Soft Voting**, where predictions are made based on class probabilities averaged over models.

**Why Voting?**  
Voting is a simpler alternative to stacking and often effective when base models are individually strong and diverse.

**Models Used:**
- XGBoost  
- Random Forest  
- LightGBM

**Voting Strategy:**
- `voting='soft'` ensures models with higher confidence contribute more to final decisions.

---

### :rocket: 3. LightGBM + Random Forest Voting (Alternative Ensemble)

This version uses **LightGBM and Random Forest** in a soft voting scheme.

**Why LightGBM?**  
LightGBM is a highly efficient gradient boosting algorithm that is particularly strong with large datasets and can reduce training time drastically.

---

### :rocket: 4. Baseline Model: Random Forest

We also used a simple **Random Forest** model as a baseline to compare performance with ensemble methods.

**Why Random Forest?**  
It is robust, handles overfitting well, and often works out-of-the-box for tabular data with minimal preprocessing.

---

## 📈 Summary

| Model Type             | Algorithms Used                     | Notes                                 |
|------------------------|-------------------------------------|----------------------------------------|
| **Stacking Classifier**| XGBoost, RandomForest → LogisticReg | Best accuracy, combines diverse models |
| **Voting Classifier**  | XGBoost + RF / LightGBM + RF        | Soft voting based on probability       |
| **Random Forest**      | Random Forest only                  | Simple and fast baseline               |
""")


smote_enabled_col, smote_enabled_description = st.columns(2)

if "smote_enabled" not in st.session_state:
    st.session_state.smote_enabled = False

with smote_enabled_col:
    if st.button("Enable SMOTE", key="enable_smote_button"):
        st.session_state.smote_enabled = True
        #st.button("Disable SMOTE", key="enable_smote_button", disabled=True)

with smote_enabled_description:
    if st.session_state.smote_enabled:
        st.write("🟢 SMOTE is enabled to handle class imbalance in the training data.")
    else:
        st.write("🔴 SMOTE is not enabled. The training data will be used as is without oversampling.")

if st.session_state.smote_enabled:
    X_train, y_train = smote_data(X_train, y_train)
    st.subheader("Label Distribution After SMOTE")
    st.bar_chart(y_train.value_counts(), use_container_width=True)
else:
    X_train, y_train = X_train, y_train

st.write("Now, we will train three different models: Stacking Model, Voting Model, and Random Forest Model. Each model will be trained on the preprocessed dataset and evaluated for its accuracy.")

if 'accuracy_df' not in st.session_state:
    st.session_state.accuracy_df = {
        "XGBoost & RF Stacking Model Accuracy": "Not run yet",
        "XGBoost + RF Voting Model Accuracy": "Not run yet",
        "Random Forest Model Accuracy": "Not run yet",
        "LightGBM & RF Voting Model Accuracy": "Not run yet"
    }

# Train models buttons
col1, col2, col3, col4 = st.columns(4)
col1_is_run, col2_is_run, col3_is_run, col4_is_run = False, False, False, False

with col1:
    if st.button("Train XGBoost + RF Stacking Model"):
        stacking_clf, y_pred, stacking_accuracy_score, stacking_rf = stacking_model.run_model(X_train, X_test, y_train, y_test)
        st.markdown(f"Test set accuracy: {stacking_accuracy_score:.2f}")
        st.session_state.accuracy_df["XGBoost & RF Stacking Model Accuracy"] = f"{stacking_accuracy_score:.2f}%"

        st.session_state.stacking_clf = stacking_clf
        st.session_state.stacking_chart_data = pd.DataFrame({
            "True Labels": pd.Series(y_test).astype(int).reset_index(drop=True),
            "Predictions": pd.Series(y_pred).astype(int)
        })
        st.session_state.stacking_is_run = True

with col2:
    if st.button("Train XGBoost + RF Voting Model"):
        voting_clf, y_pred, voting_accuracy_score, voting_rf = voting_model.run_model(X_train, X_test, y_train, y_test)
        st.markdown(f"Test set accuracy: {voting_accuracy_score:.2f}")
        st.session_state.accuracy_df["XGBoost & RF Voting Model Accuracy"] = f"{voting_accuracy_score:.2f}%"

        st.session_state.voting_clf = voting_clf
        st.session_state.voting_chart_data = pd.DataFrame({
            "True Labels": pd.Series(y_test).astype(int).reset_index(drop=True),
            "Predictions": pd.Series(y_pred).astype(int)
        })
        st.session_state.voting_is_run = True

with col3:
    if st.button("Train Random Forest Model"):
        global rf_model
        rf_model, y_pred, rf_accuracy_score = random_forest_model.run_model(X_train, X_test, y_train, y_test)
        st.markdown(f"Test set accuracy: {rf_accuracy_score:.2f}")
        st.session_state.accuracy_df["Random Forest Model Accuracy"] = f"{rf_accuracy_score:.2f}%"

        st.session_state.rf_model = rf_model
        st.session_state.rf_chart_data = pd.DataFrame({
            "True Labels": pd.Series(y_test).astype(int).reset_index(drop=True),
            "Predictions": pd.Series(y_pred).astype(int)
        })
        st.session_state.rf_is_run = True
        print(processed_df.columns.tolist())

with col4:
    if st.button("Train LightGBM + RF Voting Model"):
        voting_clf, y_pred, voting_accuracy_score = lightgbm_voting_model.run_model()

        st.markdown(f"Test set accuracy: {voting_accuracy_score:.2f}")
        st.session_state.accuracy_df["LightGBM & RF Voting Model Accuracy"] = f"{voting_accuracy_score:.2f}%"

        st.session_state.lgbm_voting_clf = voting_clf
        st.session_state.lgbm_voting_chart_data = pd.DataFrame({
            "True Labels": pd.Series(y_test).astype(int).reset_index(drop=True),
            "Predictions": pd.Series(y_pred).astype(int)
        })
        st.session_state.lgbm_voting_is_run = True

accuracy_df = pd.DataFrame.from_dict(st.session_state.accuracy_df, orient='index', columns=["Accuracy"])
st.dataframe(accuracy_df)

#### SHOW RESULTS BUTTONS ####
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("Show XGBoost + RF Stacking Model Results"):
        st.session_state.show_stacking = True
        print(processed_df.columns.tolist())

with col2:
    if st.button("Show XGBoost + RF Voting Model Results"):
        st.session_state.show_voting = True

with col3:
    if st.button("Show Random Forest Results"):
        st.session_state.show_rf = True
        
with col4:
    if st.button("Show LightGBM + RF Voting Results"):
        st.session_state.show_lgbm_voting = True

def show_classification_report(y_true, y_pred, title):
    report = classification_report(y_true, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    st.subheader(f"{title} - Classification Report")
    st.dataframe(df_report.style.format({"precision": "{:.2f}", "recall": "{:.2f}", "f1-score": "{:.2f}"}))
    
# @st.cache_data(show_spinner=False)

if st.session_state.get("show_stacking"):
    if st.session_state.get("stacking_is_run", False):
        temp_df = st.session_state.stacking_chart_data
        model = st.session_state.stacking_clf
        show_classification_report(temp_df["True Labels"], temp_df["Predictions"], "Stacking Model")
        X_sample = X_test.sample(n=100, random_state=42)
    else:
        st.warning("Train stacking model first.")

if st.session_state.get("show_voting"):
    if st.session_state.get("voting_is_run", False):
        temp_df = st.session_state.voting_chart_data
        model = st.session_state.voting_clf
        show_classification_report(temp_df["True Labels"], temp_df["Predictions"], "Voting Model")
        X_sample = X_test.sample(n=100, random_state=42)
    else:
        st.warning("Train voting model first.")

if st.session_state.get("show_rf"):
    if st.session_state.get("rf_is_run", False):
        temp_df = st.session_state.rf_chart_data
        model = st.session_state.get("rf_model")  # BÖYLE KULLAN
        show_classification_report(temp_df["True Labels"], temp_df["Predictions"], "Random Forest Model")
        X_sample = X_test.sample(n=100, random_state=42)
        
        if st.button("Show Feature Importances", key="show_feature_importances"):
            st.subheader("Feature Importances")
            feature_importances_df = feature_importances(model, X_train)
            st.dataframe(feature_importances_df)

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=feature_importances_df["importance"], y=feature_importances_df["feature"], ax=ax)
            ax.set_title("Feature Importances")
            st.pyplot(fig)
    else:
        st.warning("Train random forest model first.")
        
if st.session_state.get("show_lgbm_voting"):
    if st.session_state.get("lgbm_voting_is_run", False):
        temp_df = st.session_state.lgbm_voting_chart_data
        model = st.session_state.lgbm_voting_clf
        show_classification_report(temp_df["True Labels"], temp_df["Predictions"], "LightGBM + RF Voting Model")
        X_sample = X_test.sample(n=100, random_state=42)
    else:
        st.warning("Train LightGBM + RF Voting Model first.")


##### HEADER: Features #####
st.markdown("## 5. Features")
st.write("""
            The model uses a variety of features to make predictions about students' dropout and academic success.
            These features include students' grades, attendance, socio-economic background, and other factors that may influence their academic performance.
            The features are selected based on their relevance to the prediction task and their ability to provide meaningful insights into students' academic performance.
            The model is trained on these features to learn the patterns and relationships between the input features and the target variable (dropout or academic success).
            """)

feature_col1, feature_col2 = st.columns(2)

with feature_col1:
    st.markdown("""### Pure dataset columns (before preprocessing):""")
    st.write(df_copy.columns.tolist())
    
with feature_col2:
    st.markdown("""### Processed dataset columns (after preprocessing):""")
    st.write(processed_df.columns.tolist())


##### HEADER: Predictions #####
st.markdown("## 6. Predictions")
st.write("""
            The model allows users to make predictions about students' dropout and academic success based on the input features.
            Users can input various features related to students' academic performance and personal attributes, and the model will provide predictions about whether the student is likely to drop out or succeed academically.
            The predictions are based on the patterns learned by the model during training and are designed to provide meaningful insights into students' academic performance.
            The model's predictions can be used by educators, administrators, and policymakers to identify at-risk students and provide targeted interventions to improve their academic outcomes.
            The app provides an interactive interface for users to input features and visualize the predictions made by the model.
            """)

st.dataframe(processed_df.head())

st.markdown("### Let's make a prediction for you to see how the model works!")

# --------- ÖN HAZIRLIK (kodunuzda bir kere çalıştırılmış olacak) ----------
# from preprocess_module import preprocess_data   # örneğin
# non_processed_df, df, X_train, X_test, y_train, y_test = preprocess_data()

# Eğitilmiş model nesnelerini bir şekilde yüklediğinizi varsayalım:
# stacking_clf = load_model("stacking_model.pkl")
# voting_clf   = load_model("voting_model.pkl")
# rf_clf       = load_model("rf_model.pkl")

# Burada sadece "df" (işlenmiş DataFrame) ve "non_processed_df" (ham verinin bir kısmı) 
# kodu çalıştırmadan önce tanımlı ve erişilebilir kabul ediliyor.

st.header("Predict Student Dropout / Success")

st.write("Aşağıdaki soruları cevaplayarak, modelimiz sizin akademik durumunuzu tahmin edecek.\n"
         "Eğer bir soruyu değiştirmezseniz, o sütun için veri setindeki medyan veya mod değeri kullanılacaktır.")

# Her bir input satırı, df[col].median() veya df[col].mode()[0] kullanarak varsayılan üretiliyor.

# 1) Age at enrollment
st.subheader("1) What is your age?")
age_input = st.slider(
    "Select your age at enrollment",
    min_value=int(df["Age at enrollment"].min()),
    max_value=int(df["Age at enrollment"].max()),
    value=int(df["Age at enrollment"].mode()[0]),
    key="age_slider"
)
    
# preprocess_data() içindeki bucket’lama adımını tekrar uygula:
age_bucket_map = {
    17:0, 18:0,
    19:1,
    20:2,
    21:3, 22:3, 23:3,
    24:4, 25:4, 26:4, 27:4,
    28:5, 29:5, 30:5,
    31:6, 32:6, 33:6,
    34:7, 35:7, 36:7,
    37:8, 38:8, 39:8, 40:8, 41:8,
    42:9, 43:9, 44:9, 45:9,
    46:10, 47:10, 48:10, 49:10, 50:10,
    51:11, 52:11, 53:11, 54:11, 55:11, 57:11, 58:11, 59:11, 60:11, 61:11, 62:11, 70:11
}
mapped_age = age_bucket_map.get(int(age_input), int(df["Age at enrollment"].mode()[0]))

# 2) Marital status
st.subheader("2) What is your marital status?")
default_marital = int(df["Marital status"].mode()[0])
marital_map = {"Single": 1, "Married": 2, "Divorced": 4, "Widowed/Other": 3}
default_marital_label = [k for k, v in marital_map.items() if v == default_marital][0]
marital = st.selectbox(
    "Select one",
    options=list(marital_map.keys()),
    index=list(marital_map.keys()).index(default_marital_label)
)
mapped_marital = marital_map[marital]

# 3) Application mode
df2 = df.copy()
df2['Application mode'] = df2['Application mode'].replace({
    2: 10,
    5: 10,
    26: 10,
    27: 10,
    57: 10
})

application_mode_dict = {
    1: "1st Phase – General Contingent",
    10: "Special Admission Regime / Ordinances / Azores / International / Others",
    7: "Holders of Other Higher Courses",
    15: "International Student (Bachelor)",
    16: "1st Phase – Special Contingent (Madeira Island)",
    17: "2nd Phase – General Contingent",
    18: "3rd Phase – General Contingent",
    39: "Over 23 Years Old",
    42: "Transfer",
    43: "Change of Course",
    44: "Technological Specialization Diploma Holders",
    51: "Change of Institution/Course",
    53: "Short Cycle Diploma Holders"
}

available_codes = sorted(df2["Application mode"].unique())
available_labels = [f"{code} - {application_mode_dict.get(code, 'Other')}" for code in available_codes]

st.subheader("3) How did you apply?")

default_code = int(df2["Application mode"].mode()[0])
default_index = available_codes.index(default_code)

selected_label = st.selectbox(
    "Select your application mode",
    options=available_labels,
    index=default_index
)

mapped_app_mode = int(selected_label.split(" - ")[0])

# 4) Application order
st.subheader("4) What order was this university in your choices?")

default_order = int(df["Application order"].median())
app_order = st.slider(
    "Pick a number", 
    min_value=int(df["Application order"].min() + 1), 
    max_value=int(df["Application order"].max() + 1), 
    value=default_order,
    key="app_order_slider",
)
if app_order == 0:
    mapped_order = 1
elif app_order == 9:
    mapped_order = 5
else:
    mapped_order = app_order - 1

# 5) Course
course_dict = {
    171: "Animation and Multimedia Design",
    8014: "Social Service (evening attendance)",
    9003: "Agronomy",
    9070: "Communication Design",
    9085: "Veterinary Nursing",
    9119: "Informatics Engineering",
    9130: "Equinculture",
    9147: "Management",
    9238: "Social Service",
    9254: "Tourism",
    9500: "Nursing",
    9556: "Oral Hygiene",
    9670: "Advertising and Marketing Management",
    9773: "Journalism and Communication",
    9853: "Basic Education",
    9991: "Management (evening attendance)"
}

st.subheader("5) What is your course code?")

default_course = int(df["Course"].mode()[0])
course_options = sorted(df["Course"].unique())

course_labels = [f"{code} - {course_dict.get(code, 'Unknown')}" for code in course_options]

selected_label = st.selectbox(
    "Select the numeric course code",
    options=course_labels,
    index=course_options.index(default_course)
)

mapped_course = int(selected_label.split(" - ")[0])

# 6) Daytime/evening attendance
st.subheader("6) Was your attendance daytime or evening?")
att_map = {"Daytime": 1, "Evening": 2}
default_att = int(df["Daytime/evening attendance\t"].mode()[0])
default_att_label = [k for k, v in att_map.items() if v == default_att][0]
attendance = st.selectbox(
    "Select one",
    options=list(att_map.keys()),
    index=list(att_map.keys()).index(default_att_label)
)
mapped_attendance = att_map[attendance]

# 7) Admission grade
st.subheader("7) What is your Admission grade (0–200)?")
default_adm = float(df["Admission grade"].median())
admission_grade = st.number_input(
    "Enter a number",
    min_value=0.0,
    max_value=200.0,
    value=default_adm,
    step=0.1,
    format="%.1f"
)
mapped_admission_grade = float(admission_grade)

# 8) Displaced
st.subheader("8) Are you displaced (refugee/displaced)?")
disp_map = {"No": 0, "Yes": 1}
default_disp = int(df["Displaced"].mode()[0])
default_disp_label = [k for k, v in disp_map.items() if v == default_disp][0]
displaced = st.selectbox(
    "Select one",
    options=list(disp_map.keys()),
    index=list(disp_map.keys()).index(default_disp_label),
    key="displaced_select"
)
mapped_displaced = disp_map[displaced]

# 9) Educational special needs
st.subheader("9) Do you have educational special needs?")
ens_map = {"No": 0, "Yes": 1}
default_ens = int(df["Educational special needs"].mode()[0])
default_ens_label = [k for k, v in ens_map.items() if v == default_ens][0]
special_needs = st.selectbox(
    "Select one",
    options=list(ens_map.keys()),
    index=list(ens_map.keys()).index(default_ens_label),
    key="special_needs_select"
)
mapped_special_needs = ens_map[special_needs]

# 10) Debtor
st.subheader("10) Are you a debtor (tuition fee debt)?")
deb_map = {"No": 0, "Yes": 1}
default_deb = int(df["Debtor"].mode()[0])
default_deb_label = [k for k, v in deb_map.items() if v == default_deb][0]
debtor = st.selectbox(
    "Select one",
    options=list(deb_map.keys()),
    index=list(deb_map.keys()).index(default_deb_label),
    key ="debtor_select"
)
mapped_debtor = deb_map[debtor]

# 11) Tuition fees up to date
st.subheader("11) Are your tuition fees up to date?")
tf_map = {"No": 0, "Yes": 1}
default_tf = int(df["Tuition fees up to date"].mode()[0])
default_tf_label = [k for k, v in tf_map.items() if v == default_tf][0]
tuition = st.selectbox(
    "Select one",
    options=list(tf_map.keys()),
    index=list(tf_map.keys()).index(default_tf_label),
    key="tuition_select"
)
mapped_tuition = tf_map[tuition]

# 12) Gender
st.subheader("12) What is your gender?")
gender_map = {"Female": 0, "Male": 1}
default_gender = int(df["Gender"].mode()[0])
default_gender_label = [k for k, v in gender_map.items() if v == default_gender][0]
gender = st.selectbox(
    "Select one",
    options=list(gender_map.keys()),
    index=list(gender_map.keys()).index(default_gender_label)
)
mapped_gender = gender_map[gender]

# 13) Scholarship holder
st.subheader("13) Are you a scholarship holder?")
sch_map = {"No": 0, "Yes": 1}
default_sch = int(df["Scholarship holder"].mode()[0])
default_sch_label = [k for k, v in sch_map.items() if v == default_sch][0]
scholarship = st.selectbox(
    "Select one",
    options=list(sch_map.keys()),
    index=list(sch_map.keys()).index(default_sch_label)
)
mapped_scholarship = sch_map[scholarship]

# 14) Unemployment rate
st.subheader("14) What is the unemployment rate for your enrollment year? (0.00–1.00)")
default_unemp = float(df["Unemployment rate"].median())
unemp_rate = st.number_input(
    "Enter a decimal (e.g. 0.08):",
    min_value=0.0,
    max_value=16.2,
    value=default_unemp,
    step=0.01,
    format="%.2f"
)
mapped_unemp_rate = float(unemp_rate)

# 15) Inflation rate
st.subheader("15) Select the inflation rate category for your enrollment year:")
inflation_map = {
    "-0.8%": 0, "-0.3%": 1, "0.3%": 2, "0.5%": 3,
    "0.6%": 4, "1.4%": 5, "2.6%": 6, "2.8%": 7, "3.7%": 8
}
default_inf = int(df["Inflation rate"].mode()[0])
default_inf_label = [k for k, v in inflation_map.items() if v == default_inf][0]
inflation = st.selectbox(
    "Select one",
    options=list(inflation_map.keys()),
    index=list(inflation_map.keys()).index(default_inf_label)
)
mapped_inflation = inflation_map[inflation]

# 16) GDP
st.subheader("16) Select the GDP growth category for your enrollment year:")
gdp_map = {
    "-4.06%": 0, "-3.12%": 1, "-1.70%": 2, "-0.92%": 3,
    "0.32%": 4, "0.79%": 5, "1.74%": 6, "1.79%": 7,
    "2.02%": 8, "3.51%": 9
}
default_gdp = int(df["GDP"].mode()[0])
default_gdp_label = [k for k, v in gdp_map.items() if v == default_gdp][0]
gdp_category = st.selectbox(
    "Select one",
    options=list(gdp_map.keys()),
    index=list(gdp_map.keys()).index(default_gdp_label)
)
mapped_gdp = gdp_map[gdp_category]

# 17) sem_1_pass_rate
st.subheader("17) What is your expected 1st-semester pass rate? (0–1)")
default_s1 = float(processed_df["sem_1_pass_rate"].median())
sem_1_pass = st.slider(
    "Choose a decimal", 
    min_value=0.0, 
    max_value=1.0, 
    value=default_s1, 
    step=0.01, 
    format="%.2f",
    key="sem_1_pass_slider"
)
mapped_s1_pass = float(sem_1_pass)

# 18) sem_2_pass_rate
st.subheader("18) What is your expected 2nd-semester pass rate? (0–1)")
default_s2 = float(processed_df["sem_2_pass_rate"].median())
sem_2_pass = st.slider(
    "Choose a decimal", 
    min_value=0.0, 
    max_value=1.0, 
    value=default_s2, 
    step=0.01, 
    format="%.2f",
    key="sem_2_pass_slider"
)
mapped_s2_pass = float(sem_2_pass)

# 19) sem_1_points_per_credit
st.subheader("19) What is your 1st-semester points-per-credit ratio?")
default_s1_ppc = float(processed_df["sem_1_points_per_credit"].median())
sem_1_ppc = st.number_input(
    "Enter a decimal",
    min_value=0.0,
    max_value=10.0,
    value=default_s1_ppc,
    step=0.01,
    format="%.2f"
)
mapped_s1_ppc = float(sem_1_ppc)

# 20) sem_2_points_per_credit
st.subheader("20) What is your 2nd-semester points-per-credit ratio?")
default_s2_ppc = float(processed_df["sem_2_points_per_credit"].median())
sem_2_ppc = st.number_input(
    "Enter a decimal",
    min_value=0.0,
    max_value=10.0,
    value=default_s2_ppc,
    step=0.01,
    format="%.2f"
)
mapped_s2_ppc = float(sem_2_ppc)

# 21) sem1_success_rate
st.subheader("21) What is your 1st-semester success rate? (0–1)")
default_s1_succ = float(processed_df["sem1_success_rate"].median())
sem1_succ = st.slider(
    "Choose a decimal", 
    min_value=0.0, 
    max_value=1.0, 
    value=default_s1_succ, 
    step=0.01, 
    format="%.2f",
    key="sem1_success_rate_slider"
)
mapped_s1_succ = float(sem1_succ)

# 22) sem2_success_rate
st.subheader("22) What is your 2nd-semester success rate? (0–1)")
default_s2_succ = float(processed_df["sem2_success_rate"].median())
sem2_succ = st.slider(
    "Choose a decimal", 
    min_value=0.0, 
    max_value=1.0, 
    value=default_s2_succ, 
    step=0.01, 
    format="%.2f",
    key="sem2_success_rate_slider"
)
mapped_s2_succ = float(sem2_succ)

# 23) sem1_evaluation_rate
st.subheader("23) What is your 1st-semester evaluation rate? (0–1)")
default_s1_eval = float(processed_df["sem1_evaluation_rate"].median())
sem1_eval = st.slider(
    "Choose a decimal", 
    min_value=0.0, 
    max_value=1.0, 
    value=default_s1_eval, 
    step=0.01, 
    format="%.2f",
    key="sem1_evaluation_rate_slider"
)
mapped_s1_eval = float(sem1_eval)

# 24) sem2_evaluation_rate
st.subheader("24) What is your 2nd-semester evaluation rate? (0–1)")
default_s2_eval = float(processed_df["sem2_evaluation_rate"].median())
sem2_eval = st.slider(
    "Choose a decimal", 
    min_value=0.0, 
    max_value=1.0, 
    value=default_s2_eval, 
    step=0.01, 
    format="%.2f",
    key="sem2_evaluation_rate_slider"
)
mapped_s2_eval = float(sem2_eval)

# 25) avg_evaluations
st.subheader("25) What is your average number of evaluations across both semesters?")
default_avg_eval = float(processed_df["avg_evaluations"].median())
avg_eval = st.number_input(
    "Enter a number",
    min_value=0.0,
    max_value=100.0,
    value=default_avg_eval,
    step=1.0
)
mapped_avg_eval = float(avg_eval)

# 26–27) Father’s occupation & qualification → father_knowledge
st.subheader("26) Select your father's occupation and education level:")

father_occ_map = {
    "Unskilled": 0, "Skilled_Manual": 1, "Technical": 2,
    "Services_Sales": 3, "Administrative": 4, "Professional": 5,
    "Management": 6, "Armed_Forces": 7, "Operator_Driver": 8,
    "Other_Or_Student": 9, "Other": 10
}

default_f_occ = int(df["Father's occupation"].mode()[0])
father_occ_labels = [k for k, v in father_occ_map.items() if v == default_f_occ]
default_f_occ_label = father_occ_labels[0] if father_occ_labels else list(father_occ_map.keys())[0]

father_occupation = st.selectbox(
    "Father's occupation",
    options=list(father_occ_map.keys()),
    index=list(father_occ_map.keys()).index(default_f_occ_label)
)

father_qual_map = {"0 (Low)": 0, "1": 1, "2": 2, "3": 3, "4 (High)": 4}
default_f_qual = int(df["Father's qualification"].mode()[0])
father_qual_labels = [k for k, v in father_qual_map.items() if v == default_f_qual]
default_f_qual_label = father_qual_labels[0] if father_qual_labels else list(father_qual_map.keys())[0]

father_qualification = st.selectbox(
    "Father's qualification",
    options=list(father_qual_map.keys()),
    index=list(father_qual_map.keys()).index(default_f_qual_label)
)

mapped_father_knowledge = (father_occ_map[father_occupation] * father_qual_map[father_qualification]) / 2.0

# 28–29) Mother’s occupation & qualification → mother_knowledge
st.subheader("27) Select your mother's occupation and education level:")

mother_occ_map = {
    "Unskilled": 0, "Skilled_Manual": 1, "Technical": 2,
    "Services_Sales": 3, "Administrative": 4, "Professional": 5,
    "Management": 6, "Other_Or_Student": 7, "Other": 8
}

default_m_occ = int(df["Mother's occupation"].mode()[0])
mother_occ_labels = [k for k, v in mother_occ_map.items() if v == default_m_occ]
default_m_occ_label = mother_occ_labels[0] if mother_occ_labels else list(mother_occ_map.keys())[0]

mother_occupation = st.selectbox(
    "Mother's occupation",
    options=list(mother_occ_map.keys()),
    index=list(mother_occ_map.keys()).index(default_m_occ_label)
)

mother_qual_map = {"0 (Low)": 0, "1": 1, "2": 2, "3": 3, "4 (High)": 4}
default_m_qual = int(df["Mother's qualification"].mode()[0])
mother_qual_labels = [k for k, v in mother_qual_map.items() if v == default_m_qual]
default_m_qual_label = mother_qual_labels[0] if mother_qual_labels else list(mother_qual_map.keys())[0]

mother_qualification = st.selectbox(
    "Mother's qualification",
    options=list(mother_qual_map.keys()),
    index=list(mother_qual_map.keys()).index(default_m_qual_label)
)

mapped_mother_knowledge = (mother_occ_map[mother_occupation] * mother_qual_map[mother_qualification]) / 2.0


# 30) previous_qualification
st.subheader("28) What was your previous qualification grade and type?")
default_prev_grade = float(df["Previous qualification (grade)"].median())
prev_grade_input = st.number_input(
    "Previous qualification grade (0–200):",
    min_value=0.0,
    max_value=200.0,
    value=default_prev_grade,
    step=0.1,
    format="%.1f"
)

prev_type_map = {"Basic": 0, "Secondary": 1, "Higher": 2, "Other": 3}
default_prev_type = int(df["Previous qualification"].mode()[0])
default_prev_type_label = [k for k, v in prev_type_map.items() if v == default_prev_type][0]
prev_type_input = st.selectbox(
    "Previous qualification type",
    options=list(prev_type_map.keys()),
    index=list(prev_type_map.keys()).index(default_prev_type_label)
) 
mapped_previous_qualification = (prev_grade_input / 200.0) * prev_type_map[prev_type_input]

import pandas as pd

input_dict = {
    "Age at enrollment":            mapped_age,
    "Marital status":               mapped_marital,
    "Application mode":             mapped_app_mode,
    "Application order":            mapped_order,
    "Course":                       mapped_course,
    "Daytime/evening attendance\t": mapped_attendance,
    "Admission grade":              mapped_admission_grade,
    "Displaced":                    mapped_displaced,
    "Educational special needs":    mapped_special_needs,
    "Debtor":                       mapped_debtor,
    "Tuition fees up to date":      mapped_tuition,
    "Gender":                       mapped_gender,
    "Scholarship holder":           mapped_scholarship,
    "Unemployment rate":            mapped_unemp_rate,
    "Inflation rate":               mapped_inflation,
    "GDP":                          mapped_gdp,
    "sem_1_pass_rate":              mapped_s1_pass,
    "sem_2_pass_rate":              mapped_s2_pass,
    "sem_1_points_per_credit":      mapped_s1_ppc,
    "sem_2_points_per_credit":      mapped_s2_ppc,
    "sem1_success_rate":            mapped_s1_succ,
    "sem2_success_rate":            mapped_s2_succ,
    "sem1_evaluation_rate":         mapped_s1_eval,
    "sem2_evaluation_rate":         mapped_s2_eval,
    "avg_evaluations":              mapped_avg_eval,
    "father_knowledge":             mapped_father_knowledge,
    "mother_knowledge":             mapped_mother_knowledge,
    "previous_qualification":       mapped_previous_qualification
}

# Modelin beklediği sıralama
expected_order = ['Marital status', 'Application mode', 'Application order', 'Course',
       'Daytime/evening attendance\t', 'Admission grade', 'Displaced',
       'Educational special needs', 'Debtor', 'Tuition fees up to date',
       'Gender', 'Scholarship holder', 'Age at enrollment',
       'Unemployment rate', 'Inflation rate', 'GDP', 'previous_qualification',
       'sem_1_pass_rate', 'sem_2_pass_rate', 'sem_1_points_per_credit',
       'sem_2_points_per_credit', 'sem1_success_rate', 'sem2_success_rate',
       'sem1_evaluation_rate', 'sem2_evaluation_rate', 'avg_evaluations',
       'father_knowledge', 'mother_knowledge']

# DataFrame'e çevir ve doğru sıraya göre yeniden sırala
input_df = pd.DataFrame([input_dict])
input_df = input_df[expected_order]

st.subheader("Model Input:")
st.dataframe(input_df)

# ------------ PREDICTION ------------

def mapping_to_label(prediction):
    if prediction == 0:
        return "Dropout"
    elif prediction == 1:
        return "Enrolled"
    else:
        return "Graduate"

column1, column2, column3 = st.columns(3)

with column1:
    if st.button("Predict with Stacking Model"):
        if 'stacking_clf' in st.session_state:
            stacking_pred = st.session_state.stacking_clf.predict(input_df)[0]
            st.write(f"Stacking Model Prediction: {mapping_to_label(stacking_pred)}")
        else:
            st.warning("Stacking model is not trained yet.")
            
with column2:
    if st.button("Predict with Voting Model"):
        if 'voting_clf' in st.session_state:
            voting_pred = st.session_state.voting_clf.predict(input_df)[0]
            st.write(f"Voting Model Prediction: {mapping_to_label(voting_pred)}")
        else:
            st.warning("Voting model is not trained yet.")
            
with column3:
    if st.button("Predict with Random Forest Model"):
        if 'rf_model' in st.session_state:
            rf_pred = st.session_state.rf_model.predict(input_df)[0]
            st.write(f"Random Forest Prediction: {mapping_to_label(rf_pred)}")
        else:
            st.warning("Random Forest model is not trained yet.")  
    

st.markdown("## 7. Model Performance")
st.write("""The performance of the model is evaluated using several key metrics, including accuracy, precision, recall, and F1-score, to ensure both its effectiveness and reliability. These metrics are calculated on a test dataset that is kept separate from the training data, allowing for an unbiased assessment of the model's generalization to unseen instances.
        To gain deeper insights into the model’s predictive capabilities, performance is also visualized using confusion matrices, ROC curves, and other relevant visual tools. These visualizations help identify the strengths and potential weaknesses of the model across different classes.
        The model demonstrates a high level of accuracy, indicating its strong performance in predicting both student dropout and academic success. Additionally, the model's performance is continuously monitored and refined based on new data and user feedback, ensuring its long-term reliability and relevance.""")


st.markdown("## 8. Conclusion")
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

st.markdown("## 9. References")
st.write("""Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). Predict Students' Dropout and Academic Success [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89.""")

st.markdown("## 10. About the Author")
st.markdown("You can reach me out at [LinkedIn](https://www.linkedin.com/in/ozlemnurduman/), [GitHub](https://www.github.com/gramchelle) or [Kaggle](https://www.kaggle.com/gramchelle).")
st.markdown("Going to the stars together! :tada:")