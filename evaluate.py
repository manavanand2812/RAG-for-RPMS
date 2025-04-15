import json
from sklearn.metrics import f1_score

def normalize_answer(s):
    """Lowercase, remove punctuation/articles/extra whitespace."""
    import re
    import string
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in string.punctuation)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact(a, b):
    return int(normalize_answer(a) == normalize_answer(b))

def compute_f1(a, b):
    a_tokens = normalize_answer(a).split()
    b_tokens = normalize_answer(b).split()
    common = set(a_tokens) & set(b_tokens)
    if len(common) == 0:
        return 0.0
    precision = len(common) / len(a_tokens)
    recall = len(common) / len(b_tokens)
    return 2 * (precision * recall) / (precision + recall)

# Load expected and predicted answers
with open('expected_answers.json', 'r') as f:
    expected = json.load(f)

with open('generated_answers.json', 'r') as f:
    predicted = json.load(f)

total_exact = 0
total_f1 = 0
count = 0

for doc_id in expected:
    for i in range(len(expected[doc_id])):
        true_ans = expected[doc_id][i]
        pred_ans = predicted[doc_id][i]
        exact = compute_exact(true_ans, pred_ans)
        f1 = compute_f1(true_ans, pred_ans)

        total_exact += exact
        total_f1 += f1
        count += 1

        print(f"\nQuestion {count}:")
        print(f"Expected: {true_ans}")
        print(f"Predicted: {pred_ans}")
        print(f"Exact Match: {exact}")
        print(f"F1 Score: {f1:.2f}")

# Overall scores
print("\n--- Overall Evaluation ---")
print(f"Average Exact Match: {total_exact / count:.2f}")
print(f"Average F1 Score: {total_f1 / count:.2f}")
