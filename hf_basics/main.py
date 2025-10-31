from transformers import pipeline

def analyze_sentiments(texts):
    """
    Nimmt eine Liste von Texten und gibt Label + Score zurück.
    Behandelt Fehler sauber.
    """
    classifier = pipeline("sentiment-analysis")

    results = []
    for text in texts:
        try:
            output = classifier(text)[0]
            results.append({
                "text": text,
                "label": output["label"],
                "score": round(output["score"], 3)
            })
        except Exception as e:
            results.append({
                "text": text,
                "error": str(e)
            })
    return results


if __name__ == "__main__":
    inputs = [
        "Accenture builds amazing AI solutions.",
        "This system keeps crashing all the time.",
        "",
        "I'm not sure how I feel about this project."
    ]

    for r in analyze_sentiments(inputs):
        if "error" in r:
            print(f"⚠️ Fehler bei '{r['text']}': {r['error']}")
        else:
            print(f"{r['text']} -> {r['label']} ({r['score']})")
