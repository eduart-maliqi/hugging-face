from transformers import pipeline

classifier = pipeline("sentiment-analysis")

texts = [
    "Accenture is a great place to work.",
    "The meeting today was really boring.",
    "The weather is okay, not good, not bad."
]

results = classifier(texts)
for text, result in zip(texts, results):
    print(f"{text} -> {result['label']} ({result['score']:.2f})")

