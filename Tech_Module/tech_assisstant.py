import ollama
import os

MODEL = "phi3"


def load_document(file_path):
    if not os.path.exists(file_path):
        print("Document file not found!")
        return ""

    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def clean_answer(text):
    # Take only first sentence to avoid long explanation
    return text.split(".")[0] + "."


def ask_model(context, question):

    prompt = f"""
Answer the question using ONLY the document below.

Rules:
- Give ONLY one short sentence.
- Do NOT explain.
- Do NOT add extra information.
- If answer not found, say: "Not found in document."

Document:
{context}

Question: {question}
"""

    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": 0,
            "num_predict": 40  # limits long answers
        }
    )

    raw_answer = response["message"]["content"]
    return clean_answer(raw_answer)


def main():

    document_path = "api_guide.txt"
    context = load_document(document_path)

    if not context:
        return

    print("\n✅ Tech Assistant Ready!")
    print("Type 'exit' to quit\n")

    while True:

        q = input("Tech Question (exit to quit): ")

        if q.lower() == "exit":
            break

        answer = ask_model(context, q)

        print("\nAnswer:")
        print(answer)
        print("-" * 50)


if __name__ == "__main__":
    main()
