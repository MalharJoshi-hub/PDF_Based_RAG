import db
import retrieval
import generation

def run_full_pipeline():
    print("[1] Initializing DB and chunks...")
    chunks = db.get_all_chunks()
    lookup = {c["id"]: c for c in chunks}
    print(f"Loaded {len(chunks)} chunks.")

    print("[2] Initializing RetrieverManager...")
    mgr = retrieval.RetrieverManager()
    mgr.rebuild_all(chunks)

    print("[3] Initializing RAG Pipeline (CrossEncoder + Flan-T5)...")
    pipeline = generation.RAGPipeline(mgr)

    queries = [
        "What is Hibernate and what does it provide?",
        "What is SessionFactory in Hibernate?"
    ]

    for q in queries:
        print("\n" + "="*50)
        print(f"QUERY: {q}")
        print("="*50)

        for method in ["dense", "bm25"]:
            res = pipeline.process_query(
                query=q,
                retriever_method=method,
                top_k_retrieved=4,
                top_k_rerank=2,
                use_reranker=True,
                chunks_lookup=lookup
            )
            print(f"\n--- Method: {method.upper()} + Reranker ---")
            print(f"Answer: {res['answer']}")
            print(f"Timings: {res['timings']}")
            for cit in res["citations"]:
                print(f"  * [Citation {cit['citation_id']}] {cit['filename']} (P.{cit['page_number']}) - Rerank: {cit['rerank_score']}")

if __name__ == "__main__":
    run_full_pipeline()

