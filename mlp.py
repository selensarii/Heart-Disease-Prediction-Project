from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pandas as pd

def run_mlp():
    data_cleaned = pd.read_csv('normalized_transformed_cleaned_heart_disease.csv')
    X = data_cleaned.drop(columns=['Heart_ stroke'])
    y = data_cleaned['Heart_ stroke']

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    mlp_configs = [
        {"hidden_layer_sizes": (32,), "description": "1 gizli katman, 32 nöron"},
        {"hidden_layer_sizes": (32, 32), "description": "2 gizli katman, 32'şer nöron"},
        {"hidden_layer_sizes": (32, 32, 32), "description": "3 gizli katman, 32'şer nöron"}
    ]

    results = []
    for config in mlp_configs:
        mlp = MLPClassifier(
            hidden_layer_sizes=config["hidden_layer_sizes"],
            max_iter=2000,
            learning_rate_init=0.001,
            random_state=42,
            early_stopping=True
        )
        mlp.fit(X_train, y_train)
        y_pred = mlp.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        results.append((config["description"], accuracy, precision, recall, f1))
    return results
