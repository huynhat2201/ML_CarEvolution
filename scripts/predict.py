import pickle

def load_model(path='model/best_model.pkl'):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model

def predict(model, X_test):
    return model.predict(X_test)
