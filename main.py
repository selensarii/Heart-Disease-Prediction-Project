import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from knn import run_knn
from mlp import run_mlp
from naive_bayes import run_naive_bayes
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


try:
    data_cleaned = pd.read_csv('cleaned_heart_disease.csv')
    print("Temizlenmiş veri kümesi başarıyla yüklendi!")
except FileNotFoundError:
    print("Hata: 'cleaned_heart_disease.csv' dosyası bulunamadı. Lütfen önce bu dosyayı oluşturun.")

label_encoder = LabelEncoder()
data_cleaned['Gender'] = label_encoder.fit_transform(data_cleaned['Gender'])
data_cleaned['prevalentStroke'] = label_encoder.fit_transform(data_cleaned['prevalentStroke'])
data_cleaned['Heart_ stroke'] = label_encoder.fit_transform(data_cleaned['Heart_ stroke'])

onehot_encoder = OneHotEncoder(sparse_output=False, drop='first')
education_encoded = onehot_encoder.fit_transform(data_cleaned[['education']])
education_df = pd.DataFrame(education_encoded, columns=onehot_encoder.get_feature_names_out(['education']))
data_cleaned = pd.concat([data_cleaned, education_df], axis=1)
data_cleaned.drop(columns=['education'], inplace=True)

numerical_columns = data_cleaned.columns.drop(['Heart_ stroke'])
scaler = StandardScaler()
data_cleaned[numerical_columns] = scaler.fit_transform(data_cleaned[numerical_columns])

data_cleaned.to_csv('normalized_transformed_cleaned_heart_disease.csv', index=False)
print("\nVeri kümesi başarıyla normalize edildi ve kaydedildi.")

import pandas as pd
from knn import run_knn
from mlp import run_mlp
from naive_bayes import run_naive_bayes

if __name__ == "__main__":
    print("\nKNN Algoritması Sonuçları:")
    knn_results = run_knn()
    knn_df = pd.DataFrame(
        knn_results, 
        columns=["K Değeri", "Accuracy", "Precision", "Recall", "F1 Score"]
    )
    knn_df["Algoritma"] = "KNN"
    knn_df["Yapılandırma"] = knn_df["K Değeri"].apply(lambda x: f"K={int(x)}")
    knn_df.drop(columns=["K Değeri"], inplace=True)
    print(knn_df)

    print("\nMLP Algoritması Sonuçları:")
    mlp_results = run_mlp()
    mlp_df = pd.DataFrame(
        mlp_results, 
        columns=["Yapılandırma", "Accuracy", "Precision", "Recall", "F1 Score"]
    )
    mlp_df["Algoritma"] = "MLP"
    print(mlp_df)

    print("\nNaive Bayes Algoritması Sonuçları:")
    nb_results = run_naive_bayes()
    nb_df = pd.DataFrame(
        [nb_results], 
        columns=["Accuracy", "Precision", "Recall", "F1 Score"]
    )
    nb_df["Algoritma"] = "Naive Bayes"
    nb_df["Yapılandırma"] = "Varsayılan"
    print(nb_df)

    all_results = pd.concat([knn_df, mlp_df, nb_df], ignore_index=True)
    print("\nAlgoritmaların Başarım Değerleri Kıyaslaması P,R ve F Değerleri:")
    print(all_results)

    all_results.to_csv('algorithm_results.csv', index=False)
    print("\nAlgoritma sonuçları 'algorithm_results.csv' dosyasına kaydedildi.")
