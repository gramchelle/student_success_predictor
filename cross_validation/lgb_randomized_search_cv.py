import os
import sys
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
from scipy.stats import randint as sp_randint, uniform
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import preprocessing as pp

df, processed_df, X_train, X_test, y_train, y_test = pp.preprocess_data()

lgbm = LGBMClassifier(random_state=42)

param_dist = {
    'n_estimators': sp_randint(100, 300),
    'learning_rate': uniform(0.01, 0.3),
    'max_depth': sp_randint(3, 10),
    'num_leaves': sp_randint(15, 63),
    'subsample': uniform(0.6, 0.4)
}

random_search = RandomizedSearchCV(
    estimator=lgbm,
    param_distributions=param_dist,
    n_iter=25,
    scoring='accuracy',
    cv=3,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train)

best_model = random_search.best_estimator_
best_params = random_search.best_params_
best_score = random_search.best_score_

y_pred = best_model.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)

print("✅ Training completed!")
print("**Best Parameters:**", best_params)
print("**Best CV Accuracy:**", round(best_score, 4))
print("**Classification Report:**")
print(report)