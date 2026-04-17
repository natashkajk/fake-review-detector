import pandas as pd

df = pd.read_csv("data/dataset.csv")

df = df[["text_", "label"]].rename(columns={"text_": "text"})

# 🔥 МАППИНГ ТВОИХ ЛЕЙБЛОВ
df["label"] = df["label"].map({
    "CG": 1,   # fake (Computer Generated)
    "OR": 0    # original / real
})

# убираем мусорные строки
df = df.dropna()

df.to_csv("data/clean_dataset.csv", index=False)

print("✅ Labels converted: CG=1, OR=0")