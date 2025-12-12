import pandas as pd
from textaugment import EDA

# Load dataset
df = pd.read_csv("dev_phase/subtask3/train/eng.csv")

label_cols = [
    'stereotype', 'vilification', 'dehumanization',
    'extreme_language', 'lack_of_empathy', 'invalidation'
]

t = EDA()

# Count class sizes
class_sizes = {label: df[label].sum() for label in label_cols}
max_count = max(class_sizes.values())
print("Before balancing:", class_sizes)

augmented_rows = []

# Rotate EDA methods
def apply_eda(text, idx):
    if idx % 4 == 0:
        return t.synonym_replacement(text, n=2)
    elif idx % 4 == 1:
        return t.random_insertion(text, n=2)
    elif idx % 4 == 2:
        return t.random_swap(text, n=2)
    else:
        return t.random_deletion(text, p=0.1)

# Balance per label
for label in label_cols:
    needed = max_count - class_sizes[label]
    if needed <= 0:
        continue

    print(f"Augmenting '{label}' by {needed} samples...")

    subset = df[df[label] == 1]
    subset_len = len(subset)

    i = 0
    while needed > 0:
        row = subset.iloc[i % subset_len]
        text = row["text"]
        labels = row[label_cols].tolist()

        try:
            aug_text = apply_eda(text, i)
            new_id = f"{row['id']}_aug{i}"

            augmented_rows.append([new_id, aug_text, *labels])
            needed -= 1
        except:
            pass
        
        i += 1

columns = ["id", "text"] + label_cols
aug_df = pd.DataFrame(augmented_rows, columns=columns)

# Combine original and augmented
balanced_df = pd.concat([df, aug_df], ignore_index=True)

# Save output
balanced_df.to_csv("balanced_abusive_behaviors_eda.csv", index=False)

print("\nAfter balancing:", {label: balanced_df[label].sum() for label in label_cols})
print("\nSaved as balanced_abusive_behaviors_eda.csv ✔️")
