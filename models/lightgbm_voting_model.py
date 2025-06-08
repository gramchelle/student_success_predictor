import os
import sys
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, accuracy_score
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import preprocessing as pp

df, processed_df, X_train, X_test, y_train, y_test = pp.preprocess_data()

def run_model():
    lgbm_model = LGBMClassifier(random_state=42, learning_rate= 0.1, max_depth= 3, n_estimators= 100, num_leaves= 15, subsample= 0.6)
    rf_model = RandomForestClassifier(random_state=42, bootstrap= True, max_depth= None, min_samples_leaf= 1, min_samples_split= 5, n_estimators= 100)

    voting_clf = VotingClassifier(
        estimators=[
            ('lightgbm', lgbm_model),
            ('random_forest', rf_model)
        ],
        voting='soft'
    )

    voting_clf.fit(X_train, y_train)

    y_pred = voting_clf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    accuracy = accuracy_score(y_test, y_pred) * 100

    print("Voting Classifier is trained. ✅")
    #print("**Classification Report:**")
    #print(report)
    return voting_clf, y_pred, accuracy