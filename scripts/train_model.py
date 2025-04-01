from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import pickle
import os

def split_data(df):
    X = df.drop('class', axis=1)
    y = df['class']
    return train_test_split(X, y, test_size=0.3, random_state=42)

def train_decision_tree(X_train, y_train):
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    return clf

def train_naive_bayes(X_train, y_train):
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    return nb

def tune_decision_tree(X_train, y_train):
    param_grid = {
        'max_depth': [3, 5, 10, None],
        'criterion': ['gini', 'entropy'],
        'min_samples_split': [2, 5, 10]
    }
    grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    print("\n Best params:", grid_search.best_params_)
    print(" Best accuracy:", grid_search.best_score_)
    return grid_search.best_estimator_

def save_model(model, path='model/best_model.pkl'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f" Mô hình đã được lưu tại: {path}")
