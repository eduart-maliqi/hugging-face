from transformers import pipeline

def test_sentiment():
    classifier = pipeline("sentiment-analysis")
    text = input("Gib deinen Text ein: ")
    result = classifier(text)
    print(result)

if __name__ == "__main__":
    test_sentiment()
