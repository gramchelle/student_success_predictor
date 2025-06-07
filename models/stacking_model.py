from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from preprocessing import preprocess_data


def run_model(X_train, X_test, y_train, y_test):
    """ STACKING -- BEST SOLUTION SO FAR"""

    xgb_model = XGBClassifier(
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

    rf_model = RandomForestClassifier(random_state=42, max_depth= None, min_samples_leaf= 2, min_samples_split= 5, n_estimators= 200)

    meta_model = LogisticRegression(max_iter=1000)

    stacking_clf = StackingClassifier(
        estimators=[('xgb', xgb_model), ('rf', rf_model)],
        final_estimator=meta_model,
        cv=5,
        n_jobs=-1
    )

    stacking_clf.fit(X_train, y_train)

    y_pred = stacking_clf.predict(X_test)

    from sklearn.metrics import accuracy_score, classification_report
    accuracy = accuracy_score(y_test, y_pred) * 100
    #print(classification_report(y_test, y_pred))
    
    return stacking_clf, y_pred, accuracy

def predict(test_data, X_train, X_test, y_train, y_test):
    """ Predict using the trained stacking model """
    stacking_clf, _, _ = run_model(X_train, X_test, y_train, y_test)
    return stacking_clf.predict(test_data)

def predict_proba(test_data, X_train, X_test, y_train, y_test):
    """ Predict probabilities using the trained stacking model """
    stacking_clf, _, _ = run_model(X_train, X_test, y_train, y_test)
    return stacking_clf.predict_proba(test_data)
