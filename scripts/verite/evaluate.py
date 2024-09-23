import warnings
from multiprocessing import set_start_method

from infact.eval.evaluate import evaluate

warnings.filterwarnings("ignore")

if __name__ == '__main__':  # evaluation uses multiprocessing
    set_start_method("spawn")
    evaluate(
        llm="gpt_4o_mini",
        tools_config=dict(searcher=dict(
            search_engine_config=dict(
                google=dict(),
            ),
            limit_per_search=3
        ),
            manipulation_detector=dict(),
            object_detector=dict(),
            geolocator=dict()
        ),
        fact_checker_kwargs=dict(
            procedure_variant="summary",
            interpret=True,
            decompose=False,
            decontextualize=False,
            filter_check_worthy=False,
            max_iterations=3,
            max_result_len=64_000,  # characters
        ),
        llm_kwargs=dict(temperature=0.01),
        benchmark_name="verite",
        benchmark_kwargs=dict(variant="dev"),
        n_samples=10,
        print_log_level="info",
        random_sampling=False,
        n_workers=1,
    )
