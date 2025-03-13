from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
import numpy as np
import pandas as pd

def normalize_data(df_local, feats):
    if df_local is None or df_local.empty or not feats:
        return df_local, None
    sc = StandardScaler()
    df_local[feats] = sc.fit_transform(df_local[feats])
    return df_local, sc

def do_smote(X, y):
    try:
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X, y)
        return X_res, y_res
    except ImportError:
        return X, y

def feature_selection_rfe(X, y, base_estimator=None, n_features=5):
    if base_estimator is None:
        base_estimator = RandomForestClassifier(random_state=42)
    rfe = RFE(estimator=base_estimator, n_features_to_select=n_features)
    rfe.fit(X, y)
    support_mask = rfe.support_
    selected_features = X.columns[support_mask].tolist()
    return selected_features

def train_basic_models(X_train, y_train, X_test, y_test):
    results = {}

    # Logistic Regression
    logreg = LogisticRegression(max_iter=10000)
    logreg.fit(X_train, y_train)
    y_pred_lr = logreg.predict(X_test)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    report_lr = classification_report(y_test, y_pred_lr)
    results['LogisticRegression'] = {'model': logreg, 'accuracy': acc_lr, 'report': report_lr, 'y_pred': y_pred_lr}

    # Decision Tree
    dt = DecisionTreeClassifier(max_depth=10, random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    acc_dt = accuracy_score(y_test, y_pred_dt)
    report_dt = classification_report(y_test, y_pred_dt)
    results['DecisionTree'] = {'model': dt, 'accuracy': acc_dt, 'report': report_dt, 'y_pred': y_pred_dt}

    # Random Forest with cross validation
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='accuracy')
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    report_rf = classification_report(y_test, y_pred_rf)
    results['RandomForest'] = {'model': rf, 'accuracy': acc_rf, 'report': report_rf, 'cv_scores': cv_scores, 'cv_mean': np.mean(cv_scores), 'y_pred': y_pred_rf}

    return results

def grid_search_random_forest(X_train, y_train, X_test, y_test):
    param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 20, None], 'min_samples_split': [2, 5, 10]}
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_rf = grid_search.best_estimator_
    y_pred_rf = best_rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred_rf)
    report = classification_report(y_test, y_pred_rf)
    return best_rf, acc, report

def train_voting_ensemble(X_train, y_train, X_test, y_test, chosen_estimators):
    voting_clf = VotingClassifier(estimators=chosen_estimators, voting='soft')
    voting_clf.fit(X_train, y_train)
    y_pred = voting_clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return voting_clf, acc, report, y_pred
