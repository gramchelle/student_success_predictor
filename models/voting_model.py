from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
import xgboost as xgb
from preprocessing import preprocess_data

def run_model(X_train, X_test, y_train, y_test):
    """ VOTING """

    xgb_model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=3,
        use_label_encoder=False,
        random_state=42,
        colsample_bytree=0.8,
        learning_rate=0.2,
        max_depth=7,
        n_estimators=200,
        subsample=0.8
    )

    rf_model = RandomForestClassifier(random_state=42, n_estimators=200)

    voting_clf = VotingClassifier(
        estimators=[('xgb', xgb_model), ('rf', rf_model)],
        voting='soft',  # class-probability oylaması
        n_jobs=-1
    )

    voting_clf.fit(X_train, y_train)
    y_pred = voting_clf.predict(X_test)

    print("Voting Accuracy:", accuracy_score(y_test, y_pred))
    accuracy = accuracy_score(y_test, y_pred) * 100
    print(classification_report(y_test, y_pred))
    
    return voting_clf, y_pred, accuracy, rf_model

def predict(test_data, X_train, y_train, X_test, y_test):
    """ Predict using the trained voting model """
    voting_clf, _, _ = run_model(X_train, y_train, X_test, y_test)
    prediction = voting_clf.predict(test_data)
    return prediction
    
def predict_proba(test_data, X_train, y_train, X_test, y_test):
    """ Predict probabilities using the trained voting model """
    voting_clf, _, _ = run_model(X_train, y_train, X_test, y_test)
    return voting_clf.predict_proba(test_data)