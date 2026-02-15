import ollama
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MODEL = "phi3"

def load_doc():
    with open("issue_solutions.txt", "r", encoding="utf-8") as f:
        return [f.read()]

def retrieve(query, docs):
    vec = TfidfVectorizer()
    matrix = vec.fit_transform(docs + [query])
    return docs[0]

def ask_model(context, question):

    system_prompt = """
You are a helpdesk assistant.

RULES:
- Answer only using the support document
- Do not invent solutions
- If not found say:
  Solution not available in helpdesk document.
"""

    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Support Info:\n{context}\n\nQuestion:\n{question}"}
        ],
        options={"temperature": 0.0}
    )

    return response["message"]["content"]

def main():

    docs = load_doc()

    while True:
        q = input("\nHelpdesk Question (exit to quit): ")

        if q.lower() == "exit":
            break

        context = retrieve(q, docs)
        ans = ask_model(context, q)

        print("\nAnswer:")
        print(ans)

if __name__ == "__main__":
    main()
