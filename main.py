import pandas as pd

# Đặt tên cho các cột theo file mô tả dữ liệu
columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']

# Đọc dữ liệu trực tiếp từ file car.data trong thư mục dự án
data = pd.read_csv('car.data', names=columns)

# In ra 5 dòng đầu tiên để kiểm tra
print(data.head())

# Kiểm tra thông tin dữ liệu
print("\nThông tin về dữ liệu:")
print(data.info())

# Kiểm tra số lượng mẫu của từng nhãn (class)
print("\nPhân phối của thuộc tính mục tiêu:")
print(data['class'].value_counts())

# Hiển thị thống kê cơ bản (chỉ áp dụng cho dữ liệu dạng số)
print("\nThống kê dữ liệu:")
print(data.describe())

from sklearn.preprocessing import LabelEncoder

# Chuyển đổi dữ liệu categorical thành số
encoder = LabelEncoder()

for col in data.columns:
    data[col] = encoder.fit_transform(data[col])

print("\nDữ liệu sau khi mã hóa:")
print(data.head())

from sklearn.model_selection import train_test_split


# Phân chia tập dữ liệu (70% train - 30% test)
X = data.drop('class', axis=1)  # Tất cả các cột trừ cột 'class'
y = data['class']  # Cột mục tiêu

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print("\nKích thước tập huấn luyện:", X_train.shape)
print("Kích thước tập kiểm tra:", X_test.shape)

# Sử dụng Decision Tree để phân loại xe

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Khởi tạo mô hình
clf = DecisionTreeClassifier(random_state=42)

# Huấn luyện mô hình trên tập huấn luyện
clf.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = clf.predict(X_test)

# Đánh giá mô hình
print("\nĐộ chính xác của mô hình Decision Tree:", accuracy_score(y_test, y_pred))
print("\nBáo cáo phân loại:")
print(classification_report(y_test, y_pred))

# Sử dung mô hình Naive Bayes

from sklearn.naive_bayes import GaussianNB

# Khởi tạo mô hình Naive Bayes
nb_model = GaussianNB()

# Huấn luyện mô hình
nb_model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred_nb = nb_model.predict(X_test)

# Đánh giá mô hình
print("\nĐộ chính xác của mô hình Naive Bayes:", accuracy_score(y_test, y_pred_nb))
print("\nBáo cáo phân loại (Naive Bayes):")
print(classification_report(y_test, y_pred_nb))


# Vẽ biểu đồ phân phối dữ liệu

import seaborn as sns
import matplotlib.pyplot as plt

# # Biểu đồ phân phối của thuộc tính mục tiêu (class)
plt.figure(figsize=(8, 5))
sns.countplot(x=data['class'])
plt.title("Phân phối của nhãn (class)")
plt.xlabel("Loại xe")
plt.ylabel("Số lượng")
plt.show()


#So sánh độ chính xác của hai mô hình
accuracy_dt = accuracy_score(y_test, y_pred)  # Decision Tree
accuracy_nb = accuracy_score(y_test, y_pred_nb)  # Naive Bayes

#Tạo biểu đồ so sánh
plt.figure(figsize=(6, 4))
sns.barplot(x=["Decision Tree", "Naive Bayes"], y=[accuracy_dt, accuracy_nb])
plt.title("So sánh độ chính xác của mô hình")
plt.ylabel("Accuracy")
plt.ylim(0, 1)  # Giới hạn trục y từ 0 đến 1
plt.show()


# Vẽ cây quyết định

from sklearn.tree import plot_tree

plt.figure(figsize=(40, 100))
plot_tree(clf, feature_names=X.columns, class_names=['unacc', 'acc', 'good', 'vgood'], filled=True)
plt.title("Cấu trúc cây quyết định")
plt.show()
