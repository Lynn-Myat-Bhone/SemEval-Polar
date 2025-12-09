import pandas as pd
import random
from nltk.corpus import wordnet

# ===== CONFIG =====
TARGET_AUGMENTED = 500
# ==================

# Load your dataset
df = pd.read_csv("./dev_phase/subtask2/train/eng.csv")

label_cols = ['political', 'racial/ethnic', 'religious', 'gender/sexual', 'other']

augmented_rows = []

# ---- EDA helpers ----
def get_synonyms(word):
    syns = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            w = lemma.name().replace("_", " ").lower()
            if w != word:
                syns.add(w)
    return list(syns)

def synonym_replacement(words, n=2):
    words = words.copy()
    random.shuffle(words)
    changed = 0
    for i, w in enumerate(words):
        syns = get_synonyms(w)
        if syns:
            words[i] = random.choice(syns)
            changed += 1
        if changed >= n:
            break
    return words

def random_insertion(words, n=2):
    for _ in range(n):
        w = random.choice(words)
        syns = get_synonyms(w)
        if syns:
            pos = random.randint(0, len(words))
            words.insert(pos, random.choice(syns))
    return words

def random_swap(words, n=2):
    for _ in range(n):
        i, j = random.sample(range(len(words)), 2)
        words[i], words[j] = words[j], words[i]
    return words

def random_deletion(words, p=0.1):
    return [w for w in words if random.random() > p]

def apply_eda(text, idx):
    words = text.split()
    if len(words) < 2:
        return text
    if idx % 4 == 0:
        return " ".join(synonym_replacement(words, 2))
    elif idx % 4 == 1:
        return " ".join(random_insertion(words, 2))
    elif idx % 4 == 2:
        return " ".join(random_swap(words, 2))
    else:
        return " ".join(random_deletion(words, 0.1))

# ---- Generate ONLY 500 new samples ----
for i in range(TARGET_AUGMENTED):
    row = df.iloc[i % len(df)]

    new_text = apply_eda(row['text'], i)
    new_id = f"{row['id']}_aug{i}"
    labels = row[label_cols].tolist()

    augmented_rows.append([new_id, new_text] + labels)

# Save ONLY augmented data (no original)
aug_df = pd.DataFrame(
    augmented_rows,
    columns=['id', 'text'] + label_cols
)

aug_df.to_csv("task2_only_augmented_500.csv", index=False)

print("✅ Done — Saved ONLY new 500 rows")
print("File: task2_only_augmented_500.csv")