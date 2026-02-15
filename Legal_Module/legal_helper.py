import ollama
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MODEL = "phi3"

def load_doc():
    with open("company_contract.txt", "r", encoding="utf-8") as f:
        return [f.read()]

def retrieve(query, docs):
    vec = TfidfVectorizer()
    matrix = vec.fit_transform(docs + [query])
    return docs[0]

def ask_model(context, question):

    system_prompt = """
You are a legal assistant.

RULES:
- Answer only from contract text
- No extra explanation
- If not found say:
  Clause not found in legal document.
"""

    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Contract:\n{context}\n\nQuestion:\n{question}"}
        ],
        options={"temperature": 0.0}
    )

    return response["message"]["content"]

def main():

    docs = load_doc()

    while True:
        q = input("\nLegal Question (exit to quit): ")

        if q.lower() == "exit":
            break

        context = retrieve(q, docs)
        ans = ask_model(context, q)

        print("\nAnswer:")
        print(ans)

if __name__ == "__main__":
    main()
