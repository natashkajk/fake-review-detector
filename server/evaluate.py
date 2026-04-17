import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# загрузка данных
df = pd.read_csv("data/clean_dataset.csv")

#import numpy as np
from sklearn.metrics import accuracy_score
from train import trainer, train_dataset, test_dataset

# TEST accuracy
predictions = trainer.predict(test_dataset)

preds = np.argmax(predictions.predictions, axis=1)
labels = predictions.label_ids

test_acc = accuracy_score(labels, preds)
print("Test Accuracy:", test_acc)

# TRAIN accuracy
train_preds = trainer.predict(train_dataset)

train_preds_labels = np.argmax(train_preds.predictions, axis=1)
train_labels = train_preds.label_ids

train_acc = accuracy_score(train_labels, train_preds_labels)
print("Train Accuracy:", train_acc)