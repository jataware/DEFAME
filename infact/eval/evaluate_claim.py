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
    tools     = initialize_tools(tools_config, logger=logger, device=device)
    llm       = model_specifier_to_shorthand(llm) if llm not in AVAILABLE_MODELS["Shorthand"].values else llm
    llm_model = make_model(llm, logger=logger, device=device, **llm_kwargs)

    # Setup fact checker
    fc = FactChecker(
        llm     = llm_model,
        tools   = tools,
        logger  = logger,
        **(fact_checker_kwargs or {})
    )

    results = []
    all_instance_stats = pd.DataFrame()
    start_time = time.time()

    # Process each claim
    for idx, claim_text in enumerate(claims):
        _, docs, metas = fc.check_content(claim_text)
        doc = docs[0]
        meta = metas[0]
        results.append({
            "id": idx,
            "claim": claim_text,
            "verdict": doc.verdict,
            "justification": doc.justification
        })    

    return results


