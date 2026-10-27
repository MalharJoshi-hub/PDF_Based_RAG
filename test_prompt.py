import db
import retrieval
import generation

def run_prompt_comparison():
    chunks = db.get_all_chunks()
    if not chunks:
        print("No chunks found in database.")
        return
    lookup = {c["id"]: c for c in chunks}
    mgr = retrieval.RetrieverManager()
    mgr.rebuild_all(chunks)

    q = "What is Hibernate and what does it provide?"
    dense_hits = mgr.search(q, method="dense", top_k=4, chunks_lookup=lookup)["results"]
    reranker = generation.NeuralReranker()
    top_chunks, _ = reranker.rerank(q, dense_hits, top_k=2)

    generator = generation.FlanT5Generator()

    # Compare two prompt templates:
    prompts = [
        # Template 1
        (
        "Summarization Prompt",
        f"Context:\n" + "\n\n".join([c["text"][:400] for c in top_chunks]) + f"\n\nQuestion: {q}\nSummarize the answer clearly in 2-3 sentences based on the context:"
    ),
    # Template 2
    (
        "Direct QA Prompt",
        f"Read this context:\n" + "\n\n".join([c["text"][:400] for c in top_chunks]) + f"\n\nWhat is the explanation for: {q}?"
    ),
    # Template 3
    (
        "Instruction QA Prompt",
        f"Context information:\n" + "\n\n".join([c["text"][:350] for c in top_chunks]) + f"\n\nAnswer the question using the context above: {q}"
    )
]

    for name, p in prompts:
        inputs = generator.tokenizer(p, return_tensors="pt", truncation=True, max_length=1024)
        outputs = generator.model.generate(
            **inputs,
            max_new_tokens=150,
            min_length=20,
            num_beams=3,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3
        )
        ans = generator.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"=== {name} ===")
        print(ans)
        print()

if __name__ == "__main__":
    run_prompt_comparison()

