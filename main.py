import pandas as pd
from deeptables.models.deeptable import DeepTable, ModelConfig
from deeptables.models.deepnets import WideDeep, fm_nets, dnn_nets
from deeptables.datasets import dsutils

from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score, accuracy_score
from sklearn.preprocessing import KBinsDiscretizer

from deeptables.models.deepnets import DeepFM


filename = 'IDSAI.csv'
df_IDSAI = pd.read_csv(filename)
print(df_IDSAI.shape)
features = df_IDSAI.copy()
features = features.drop(['label', 'tipo_ataque'], axis=1)
features = features.drop(['ip_src', 'ip_dst', 'port_src', 'port_dst', 'protocols'], axis=1)
labels = df_IDSAI.copy()
labels_binary = labels['label'].values
labels_multiclass = labels['tipo_ataque'].values
print(labels_binary)
print(labels_multiclass)
X_test = features
y_test = labels_binary
conf = ModelConfig(
    nets=DeepFM,
    categorical_columns='auto',
    metrics=['AUC', 'accuracy'],
    auto_categorize=True,
    auto_discrete=False,
    embeddings_output_dim=20,
    embedding_dropout=0.3,
    earlystopping_patience=9999
    )
dt = DeepTable(config=conf)
dt.fit(X_test, y_test, epochs=1)

preds = dt.predict(X_test)
precision = precision_score(y_test, preds)

recall = recall_score(y_test, preds)

f1 = f1_score(y_test, preds)


conf_matrix = confusion_matrix(y_test, preds)
accuracy_score = accuracy_score(y_test, preds)
# Print the results

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:\n", conf_matrix)
print("accuracy_score:", accuracy_score)