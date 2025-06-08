#from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
#from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE


def preprocess_data():
    # fetch dataset
    #predict_students_dropout_and_academic_success = fetch_ucirepo(id=697)

    # data (as pandas dataframes)
    #X = predict_students_dropout_and_academic_success.data.features
    #y = predict_students_dropout_and_academic_success.data.targets

    #df = X.join(y)
    
    df = pd.read_csv("C://Users//Özlem Nur//Desktop//students_dropout_st_model//data.csv", sep=";")

    non_processed_df = df.copy()
    
    le = LabelEncoder()
    
    df["Target"] = le.fit_transform(df["Target"])

    df["Application order"].value_counts()

    df["Application order"] = df["Application order"].replace({0: 1, 9: 5})

    df["Age at enrollment"] = df["Age at enrollment"].map({17: 0, 18: 0,
                                                        19: 1, 20: 2,
                                                        21: 3, 22: 3, 23: 3,
                                                        24: 4,25: 4,26: 4,27: 4,
                                                        28: 5,29: 5,30: 5,
                                                        31: 6,32: 6,33: 6,
                                                        34: 7,35: 7,36: 7,
                                                        37: 8,38: 8,39: 8,40: 8, 41:8,
                                                        42: 9,43: 9,44: 9,45: 9,
                                                        46: 10,47: 10,48: 10,49: 10,50: 10,
                                                        51: 11,52: 11,53: 11,54: 11,55: 11,57: 11,58: 11,59: 11,60: 11,61: 11,62: 11,70: 11})

    df["Marital status"] = df["Marital status"].replace(6, 4)
    df["Marital status"] = df["Marital status"].replace(3, 4)

    df['Application mode'] = df['Application mode'].replace({2: 10, 57:10, 26: 10, 27: 10, 5:10})

    df["Course"] = df["Course"].replace(33, 9556)

    def simplify_qualification(x):
        if x == 1:
            return 'Secondary'
        elif x in [2, 3, 4, 5, 6, 39, 40, 42, 43]:
            return 'Higher'
        elif x in [9, 10, 12, 14, 15, 19, 38]:
            return 'Basic'
        else:
            return 'Other'

    df['Previous qualification'] = df['Previous qualification'].apply(simplify_qualification)

    df["Previous qualification"] = df["Previous qualification"].replace({"Basic": "0", "Secondary": "1", "Higher": "2"}).astype(int)

    df["previous_qualification"] = (df["Previous qualification (grade)"] / 200) * df["Previous qualification"]
    df = df.drop(columns=["Previous qualification (grade)", "Previous qualification"])

    df['Nacionality'] = df['Nacionality'].apply(lambda x: 1 if x == 1 else 0)
    df = df.drop("Nacionality", axis=1)

    def group_mother_qualification(x):
        if x in [2, 3, 4, 5, 6, 39, 40, 41, 42, 43, 44]:
            return '3'
        elif x in [1, 9, 12, 14, 18, 22, 27, 29]:
            return '2'
        elif x in [10, 11, 19, 26, 30, 36, 37, 38]:
            return '1'
        elif x in [34, 35]:
            return '0'
        else:
            return '4'

    df["Mother's qualification"] = df["Mother's qualification"].apply(group_mother_qualification)
    df["Mother's qualification"] = df["Mother's qualification"].astype(int)

    def group_father_qualification(x):
        if x in [2, 3, 4, 5, 6, 39, 40, 41, 42, 43, 44]:
            return '3'
        elif x in [1, 9, 12, 13, 14, 18, 20, 22, 25, 27, 29, 31, 33]:
            return '2'
        elif x in [10, 11, 19, 26, 30, 36, 37, 38]:
            return '1'
        elif x in [34, 35]:
            return '0'
        else:
            return '4'

    df["Father's qualification"] = df["Father's qualification"].apply(group_father_qualification)
    df["Father's qualification"] = df["Father's qualification"].astype(int)

    def group_mother_occupation(x):
        if x in [9, 191, 192, 193, 194]:
            return 'Unskilled'
        elif x in [6, 7, 171, 173, 175]:
            return 'Skilled_Manual'
        elif x in [3, 131, 132, 134, 143, 144]:
            return 'Technical'
        elif x in [5, 151, 152, 153]:
            return 'Services_Sales'
        elif x in [4, 141]:
            return 'Administrative'
        elif x in [2, 122, 123, 125]:
            return 'Professional'
        elif x == 1:
            return 'Management'
        elif x in [0, 8, 10, 90, 99]:
            return 'Other_or_Student'
        else:
            return 'Other'

    df["Mother's occupation"] = df["Mother's occupation"].apply(group_mother_occupation)
    df["Mother's occupation"] = le.fit_transform(df["Mother's occupation"])

    #df["Father's occupation"].value_counts()

    def group_father_occupation(x):
        if x in [9, 192, 193, 194, 195]:
            return 'Unskilled'
        elif x in [6, 7, 171, 172, 174, 175, 163, 161]:
            return 'Skilled_Manual'
        elif x in [3, 131, 132, 134, 135, 143, 144]:
            return 'Technical'
        elif x in [5, 151, 152, 153, 154]:
            return 'Services_Sales'
        elif x in [4, 141]:
            return 'Administrative'
        elif x in [2, 122, 123, 124, 121]:
            return 'Professional'
        elif x in [1, 112, 114]:
            return 'Management'
        elif x in [10, 101, 102, 103]:
            return 'Armed_Forces'
        elif x in [8, 181, 182, 183]:
            return 'Operator_Driver'
        elif x in [0, 90, 99]:
            return 'Other_or_Student'
        else:
            return 'Other'

    df["Father's occupation"] = df["Father's occupation"].apply(group_father_occupation)
    df["Father's occupation"] = le.fit_transform(df["Father's occupation"])


    features = [
        'Displaced',
        'Educational special needs', ## TBD
        'Debtor',
        'Tuition fees up to date',
        'Gender',
        'Scholarship holder'
    ]
    target = 'Target'

    continuous_features = ['Admission grade', 'Age at enrollment']

    df = df.drop("International", axis=1)

    df["Inflation rate"] = df["Inflation rate"].replace({-0.8 : 0,
                                                        -0.3 : 1,
                                                        0.3 : 2,
                                                        0.5 : 3,
                                                        0.6 : 4,
                                                        1.4 : 5,
                                                        2.6 : 6,
                                                        2.8: 7,
                                                        3.7 : 8
                                                        }).astype(int)

    #df["GDP"].value_counts()

    df["GDP"] = df["GDP"].replace({
        -4.06: 0,
        -3.12: 1,
        -1.70: 2,
        -0.92: 3,
        0.32: 4,
        0.79: 5,
        1.74: 6,
        1.79: 7,
        2.02: 8,
        3.51: 9
    }).astype(int)

    Q1 = df['Admission grade'].quantile(0.25)
    Q3 = df['Admission grade'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df['Admission grade'] >= lower_bound) & (df['Admission grade'] <= upper_bound)]


    curricular_units_columns = [
        "Curricular units 1st sem (credited)",
        "Curricular units 1st sem (evaluations)",
        "Curricular units 1st sem (grade)",
        "Curricular units 1st sem (enrolled)",
        "Curricular units 1st sem (approved)",
        "Curricular units 1st sem (without evaluations)",
        "Curricular units 2nd sem (credited)",
        "Curricular units 2nd sem (evaluations)",
        "Curricular units 2nd sem (grade)",
        "Curricular units 2nd sem (enrolled)",
        "Curricular units 2nd sem (approved)",
        "Curricular units 2nd sem (without evaluations)"
    ]

    df["sem_1_pass_rate"] = df["Curricular units 1st sem (approved)"] /  df["Curricular units 1st sem (enrolled)"].replace(0, 1)
    df["sem_2_pass_rate"] = df["Curricular units 2nd sem (approved)"] /  df["Curricular units 2nd sem (enrolled)"].replace(0, 1)

    df["sem_1_points_per_credit"] = df["Curricular units 1st sem (grade)"] /  df["Curricular units 1st sem (approved)"].replace(0, 1)
    df["sem_2_points_per_credit"] = df["Curricular units 2nd sem (grade)"] /  df["Curricular units 2nd sem (approved)"].replace(0, 1)

    df['sem1_success_rate'] = df['Curricular units 1st sem (credited)'] / df['Curricular units 1st sem (enrolled)'].replace(0, 1)
    df['sem2_success_rate'] = df['Curricular units 2nd sem (credited)'] / df['Curricular units 2nd sem (enrolled)'].replace(0, 1)

    df['sem1_evaluation_rate'] = (df['Curricular units 1st sem (enrolled)'] - df['Curricular units 1st sem (without evaluations)']) / df['Curricular units 1st sem (enrolled)'].replace(0, 1)
    df['sem2_evaluation_rate'] = (df['Curricular units 2nd sem (enrolled)'] - df['Curricular units 2nd sem (without evaluations)']) / df['Curricular units 2nd sem (enrolled)'].replace(0, 1)

    df["avg_evaluations"] = (df["Curricular units 1st sem (evaluations)"] + df["Curricular units 2nd sem (evaluations)"]) / 2

    df = df.drop(columns=curricular_units_columns, axis=1)

    df["father_knowledge"] = df["Father's occupation"] * df["Father's qualification"] / 2
    df["mother_knowledge"] = df["Mother's occupation"] * df["Mother's qualification"] / 2

    df = df.drop(columns= ["Father's occupation", "Father's qualification", "Mother's occupation", "Mother's qualification"], axis=1)

    """# SMOTE"""

    # SMOTEENN: Bu kombinasyon sınıf dengesizliğini azaltmada etkili oldu. Sınıf 1 gibi az görülen sınıfların performansı arttı, bu da dengesiz sınıf problemlerinde beklenen bir başarı göstergesidir.

    label = "Target"
    X = df.drop(label, axis=1)
    y = df[label]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    print(X_train.columns)
    return non_processed_df, df, X_train, X_test, y_train, y_test

#df, processed_df, X_train, X_test, y_train, y_test = preprocess_data()
#print(y_train.shape)
#print(y_test.shape)

def smote_data(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled


def feature_importances(model, X_train):
    importances = model.feature_importances_
    feature_names = X_train.columns

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False)

    print(importance_df)
    return importance_df