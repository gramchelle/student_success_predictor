from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def run_model(X_train, X_test, y_train, y_test):
    
    #model = RandomForestClassifier(random_state=42)#, max_depth= None, min_samples_leaf= 2, min_samples_split= 5, n_estimators= 200)
    model = RandomForestClassifier(random_state=42, max_depth= None, min_samples_leaf= 2, min_samples_split= 5, n_estimators= 200)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = 100 * accuracy_score(y_test, y_pred)
    #print(classification_report(y_test, y_pred))
    
    return model, y_pred, accuracy

def predict(test_data, X_train, X_test, y_train, y_test):
    """ Predict using the trained random forest model """
    rf_model, _, _ = run_model(X_train, X_test, y_train, y_test)
    prediction = rf_model.predict(test_data)
    return prediction