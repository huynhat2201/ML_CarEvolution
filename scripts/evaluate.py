from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree

def evaluate_model(model, X_test, y_test, name="Model"):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n {name} Accuracy: {acc}")
    print(classification_report(y_test, y_pred))
    return acc

def plot_class_distribution(data):
    plt.figure(figsize=(8, 5))
    sns.countplot(x=data['class'])
    plt.title("Phân phối của nhãn (class)")
    plt.xlabel("Loại xe")
    plt.ylabel("Số lượng")
    plt.show()

def plot_accuracy_comparison(acc_dt, acc_nb):
    models = ["Decision Tree", "Naive Bayes"]
    scores = [acc_dt, acc_nb]
    plt.figure(figsize=(6, 4))
    sns.barplot(x=models, y=scores)
    plt.title("So sánh độ chính xác của mô hình")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.show()

def plot_tree_structure(model, feature_names):
    plt.figure(figsize=(40, 100))
    plot_tree(model, feature_names=feature_names, class_names=['unacc', 'acc', 'good', 'vgood'], filled=True)
    plt.title("Cấu trúc cây quyết định")
    plt.show()
