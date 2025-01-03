import time
from typing import Sequence, Optional
from pathlib import Path
from queue import Queue

import pandas as pd
import torch
import numpy as np

from infact.common.label import Label
from infact.common.modeling import model_specifier_to_shorthand, AVAILABLE_MODELS, Model, make_model
from infact.common.logger import Logger
from infact.fact_checker import FactChecker
from infact.tools import initialize_tools, Searcher
from infact.utils.utils import flatten_dict
from infact.tools.search.knowledge_base_arxiv import ArxivKnowledgeBase

def evaluate_claim(
        claims: Sequence[str],
        llm: str,
        tools_config: dict,
        fact_checker_kwargs: dict = None,
        llm_kwargs: dict = None,
        print_log_level: str = "info",
        device: Optional[str] = None,
        data_base_dir: str = "data/"
    ):
    """Evaluates claims using ArXiv knowledge base."""
    
    if llm_kwargs is None:
        llm_kwargs = dict()

    # Convert model name if needed
    procedure_variant = None if fact_checker_kwargs is None else fact_checker_kwargs.get("procedure_variant")
    
    logger = Logger(
        benchmark_name="claim_evaluation",
        procedure_name=procedure_variant,
        model_name=llm,
        print_log_level=print_log_level
    )

    # Setup device
    if device is None:
        device = f"cuda:0" if torch.cuda.is_available() else "cpu"

    # Initialize tools and model
    print(tools_config)
    tools = initialize_tools(tools_config, logger=logger, device=device)
    
    print(tools)
    llm       = model_specifier_to_shorthand(llm) if llm not in AVAILABLE_MODELS["Shorthand"].values else llm
    llm_model = make_model(llm, logger=logger, device=device, **llm_kwargs)

    # Setup fact checker
    fc = FactChecker(
        llm=llm_model,
        tools=tools,
        logger=logger,
        **(fact_checker_kwargs or {})
    )

    results = []
    all_instance_stats = pd.DataFrame()
    start_time = time.time()

    # Process each claim
    for idx, claim_text in enumerate(claims):
        logger.set_current_fc_id(idx)
        print('0')
        _, docs, metas = fc.check_content(claim_text)
        print('1')
        doc = docs[0]
        meta = metas[0]
        print('2')
        # Handle statistics
        instance_stats = flatten_dict(meta["Statistics"])
        instance_stats["ID"] = idx
        instance_stats = pd.DataFrame([instance_stats])
        all_instance_stats = pd.concat([all_instance_stats, instance_stats], ignore_index=True)
        print('3')
        results.append({
            "id": idx,
            "claim": claim_text,
            "verdict": doc.verdict,
            "justification": doc.justification
        })

        logger.save_fc_doc(doc, idx)

    duration = time.time() - start_time
    
    # Aggregate statistics similar to evaluate.py
    stats = {
        "Total run duration": duration,
        "Time per claim": all_instance_stats["Duration"].mean(),
    }
    stats.update(aggregate_stats(all_instance_stats, category="Model"))
    stats.update(aggregate_stats(all_instance_stats, category="Tools"))

    # Save statistics
    all_instance_stats.index = all_instance_stats["ID"]
    all_instance_stats.to_csv(logger.target_dir / "instance_stats.csv")

    return results, stats

def aggregate_stats(instance_stats: pd.DataFrame, category: str) -> dict[str, float]:
    """Sums the values for all instances for all the columns the name of
    which begin with 'category'."""
    aggregated_stats = dict()
    columns = list(instance_stats.columns)
    for column in columns:
        if column.startswith(category):
            aggregated = instance_stats[column].sum()
            if isinstance(aggregated, np.integer):
                aggregated = int(aggregated)
            elif isinstance(aggregated, np.floating):
                aggregated = float(aggregated)
            aggregated_stats[column] = aggregated
    return unroll_dict(aggregated_stats)

