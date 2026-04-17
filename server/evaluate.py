import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
import numpy as np
from sklearn.metrics import accuracy_score

# загрузка данных
df = pd.read_csv("data/clean_dataset.csv")

# можно взять часть данных для проверки
df = df.sample(5000)

dataset = Dataset.from_pandas(df)

# загрузка модели
model = AutoModelForSequenceClassification.from_pretrained("./model")
tokenizer = AutoTokenizer.from_pretrained("./model")

# токенизация
def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.remove_columns(["text"])
dataset.set_format("torch")

# метрика
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return {"accuracy": accuracy_score(labels, predictions)}

# Trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# оценка
results = trainer.evaluate(dataset)

print(" Accuracy:", results["eval_accuracy"])