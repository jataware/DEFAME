from infact.eval.evaluate_claim import evaluate_claim

claims = [
    # "The transformer architecture was first introduced in 2017",
    # "BERT is based on the transformer architecture",
    "Is there an mRNA cancer vaccine?"
]

if __name__ == '__main__':
    results, stats = evaluate_claim(
        claims=claims,
        llm="gpt_4o",
        tools_config=dict(
            searcher=dict(
                search_engine_config=dict(
                    arxiv_kb=dict()  # Empty dict for default ArxivKB settings
                ),
                limit_per_search=5,
            )
        ),
        fact_checker_kwargs=dict(
            procedure_variant="infact",
            max_iterations=3,
            max_result_len=64_000,  # characters
        ),
        llm_kwargs=dict(temperature=0.01),
        print_log_level="info",
        data_base_dir="data/"  # Path to where your arxiv data is stored
    )

    # Print results
    for result in results:
        print(f"\nClaim: {result['claim']}")
        print(f"Verdict: {result['verdict']}")
        print(f"Justification: {result['justification']}")

    # Print statistics
    print("\nStatistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")