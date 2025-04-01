import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(filepath='data/car.data'):
    columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
    df = pd.read_csv(filepath, names=columns)

    print(" 5 dòng đầu tiên:")
    print(df.head())

    print("\n Thông tin dữ liệu:")
    print(df.info())

    print("\n Phân phối class:")
    print(df['class'].value_counts())

    print("\n Thống kê:")
    print(df.describe())

    encoder = LabelEncoder()
    for col in df.columns:
        df[col] = encoder.fit_transform(df[col])

    print("\n Dữ liệu sau khi mã hóa:")
    print(df.head())

    return df
