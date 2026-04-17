import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", device)

# 1. модель и токенизатор
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. функция токенизации
def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

# 3. загрузка датасета
df = pd.read_csv("data/clean_dataset.csv")

# берём подвыборку для ускорения
df = df.sample(15000, random_state=42)

# оставляем нужные колонки
df = df[["text", "label"]]

# 4. train/test split (ЧЕСТНЫЙ)
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# 5. Dataset HF
train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
test_dataset = Dataset.from_pandas(test_df, preserve_index=False)

# 6. токенизация
train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

# убираем лишнюю колонку
train_dataset = train_dataset.remove_columns(["text"])
test_dataset = test_dataset.remove_columns(["text"])

train_dataset.set_format("torch")
test_dataset.set_format("torch")

# 7. модель
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
).to(device)

# 8. обучение
training_args = TrainingArguments(
    output_dir="./model",
    num_train_epochs=1,              # быстрое обучение
    per_device_train_batch_size=32,  # быстрее чем 8
    per_device_eval_batch_size=32,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    report_to="none",
    fp16=True,
    remove_unused_columns=False,
    dataloader_num_workers=0
)

# 9. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# 10. обучение
trainer.train()

# 11. сохранение модели
model.save_pretrained("./model")
tokenizer.save_pretrained("./model")

print(" Model trained and saved!")