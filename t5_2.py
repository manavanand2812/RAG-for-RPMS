import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# --- Load preprocessed documents ---
with open('documents.json', 'r') as doc_file:
    preprocessed_documents = json.load(doc_file)

documents_text = [' '.join(doc) for doc in preprocessed_documents]

# --- Load questions ---
with open('questions.json', 'r') as q_file:
    question_dict = json.load(q_file)

# Flatten questions
all_questions = []
for doc_label, questions in question_dict.items():
    for q in questions:
        all_questions.append((q, doc_label))

# --- TF-IDF setup ---
vectorizer = TfidfVectorizer()
tfidf_documents = vectorizer.fit_transform(documents_text)

def retrieve_most_relevant_doc(question):
    tfidf_question = vectorizer.transform([question])
    similarity_scores = cosine_similarity(tfidf_question, tfidf_documents)
    most_relevant_index = similarity_scores.argmax()
    return most_relevant_index, similarity_scores.max()

# --- Load T5 Model & Tokenizer ---
tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained('t5-base')

# --- Answer Generation Function ---
def generate_answer(document_text, question):
    prompt = f"question: {question} context: {document_text}"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs.input_ids, max_length=100, num_beams=4, early_stopping=True)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# --- Generate and Store Answers ---
generated_answers = {}

for idx, (question, original_doc_label) in enumerate(all_questions):
    doc_index, score = retrieve_most_relevant_doc(question)
    document = documents_text[doc_index]
    answer = generate_answer(document, question)

    print(f"\nQ{idx+1}: {question}")
    print(f"Original Group: {original_doc_label}")
    print(f"→ Retrieved Document: Document {doc_index + 1} (Score: {score:.4f})")
    print(f"→ Answer: {answer}")

    # Save the answer grouped by document
    if original_doc_label not in generated_answers:
        generated_answers[original_doc_label] = []
    generated_answers[original_doc_label].append(answer)

# --- Save to JSON file ---
with open('generated_answers.json', 'w') as out_file:
    json.dump(generated_answers, out_file, indent=4)

print("\n✅ Answers saved to generated_answers.json")
