from infact.eval.evaluate_claim import evaluate_claim

claims = [
    # "The transformer architecture was first introduced in 2017",
    # "BERT is based on the transformer architecture",
    # "Is there an mRNA cancer vaccine?"
    # "Walkable Neighborhoods help adults socialize, increase community",
    "Electromagnetic radiation by mobile telephone masts may cause disease in mamals"
]


high_performance_concrete_claims = [
    "The incorporation of steel fibers in UHPC only significantly improves tensile strength when fiber length exceeds 12mm and the fiber aspect ratio is greater than 65, but this relationship breaks down when silica fume content exceeds 25% of total binder weight",
    "Nano-titanium dioxide inclusion in UHPC matrices increases early-age compressive strength development only in the presence of supplementary cementitious materials with high alumina content, and this effect is maximized at 3% TiO2 by weight of cement",
    "The autogenous shrinkage reduction achieved by internal curing with superabsorbent polymers in UHPC is only effective when the polymer absorption capacity exceeds 30g/g and initial water-to-binder ratio is below 0.20, contrary to conventional understanding of internal curing mechanisms",
    "Steam curing of UHPC at temperatures above 90°C can actually decrease long-term durability when the mix contains high volumes of ground granulated blast furnace slag (>40% replacement), due to altered C-S-H gel formation and microstructural reorganization",
    "The incorporation of basalt fibers in UHPC can achieve equivalent mechanical properties to steel fiber reinforcement only when the matrix contains both silica fume and metakaolin in specific proportions (15% and 10% respectively of total binder content) and when fiber length distribution is bimodal",
    "The beneficial effects of graphene oxide on UHPC strength development are significantly dependent on the oxidation degree of the GO sheets, with optimal performance occurring only when the C/O ratio is between 2.1 and 2.4, and these benefits diminish in mixes with high calcium aluminate cement content"
]
# Answers:  Highly Feasible, Moderately Feasible, Highly Feasible, Very Highly Feasible, Less Feasible, Not Feasible



if __name__ == '__main__':
    results = evaluate_claim(
        claims=high_performance_concrete_claims,
        llm="gpt_4o",
        tools_config=dict(
            searcher=dict(
                search_engine_config=dict(
                    #arxiv_kb=dict()
                    materials_kb=dict()
                ),
                limit_per_search=5,
            )
        ),
        fact_checker_kwargs=dict(
            # procedure_variant="infact",
            procedure_variant="advanced",
            max_iterations=3,
            max_result_len=64_000,  # characters
        ),
        llm_kwargs=dict(temperature=0.01),
        print_log_level="info",
        data_base_dir="data/"  
    )

    # Print results
    for result in results:
        print(f"\nClaim: {result['claim']}")
        print(f"Verdict: {result['verdict']}")
        print(f"Justification: {result['justification']}")
