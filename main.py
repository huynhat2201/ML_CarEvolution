from scripts.load_data import load_and_preprocess_data
from scripts.train_model import (
    split_data, train_decision_tree, train_naive_bayes,
    tune_decision_tree, save_model
)
from scripts.evaluate import (
    evaluate_model, plot_class_distribution,
    plot_accuracy_comparison, plot_tree_structure
)
from scripts.predict import load_model, predict

# 1. Load dữ liệu & xử lý
df = load_and_preprocess_data()

# 2. Trực quan hóa phân phối nhãn
plot_class_distribution(df)

# 3. Chia dữ liệu
X_train, X_test, y_train, y_test = split_data(df)

# 4. Huấn luyện hai mô hình
dt_model = train_decision_tree(X_train, y_train)
nb_model = train_naive_bayes(X_train, y_train)

# 5. Đánh giá mô hình
acc_dt = evaluate_model(dt_model, X_test, y_test, "Decision Tree")
acc_nb = evaluate_model(nb_model, X_test, y_test, "Naive Bayes")

# 6. Biểu đồ so sánh
plot_accuracy_comparison(acc_dt, acc_nb)

# 7. Vẽ cây quyết định
plot_tree_structure(dt_model, X_train.columns)

# 8. Tối ưu cây quyết định với GridSearchCV
best_model = tune_decision_tree(X_train, y_train)

# 9. Lưu mô hình tốt nhất
save_model(best_model)

# 10. Load và dự đoán với mô hình đã lưu
loaded_model = load_model()
y_pred_loaded = predict(loaded_model, X_test)

from sklearn.metrics import accuracy_score
print("\n Accuracy mô hình đã lưu:", accuracy_score(y_test, y_pred_loaded))
