import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# 1. загружаем датасет
df = pd.read_csv("data/clean_dataset.csv")

# проверка колонок
df = df[["text", "label"]]

dataset = Dataset.from_pandas(df)

# 2. модель
model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

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

# 3. загружаем модель
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)

# 4. настройки обучения
training_args = TrainingArguments(
    output_dir="./model",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    logging_steps=50,
    save_strategy="epoch"
)

# 5. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# 6. обучение
trainer.train()

# 7. сохраняем модель
model.save_pretrained("./model")
tokenizer.save_pretrained("./model")

print("✅ Model trained and saved!")