import time
from typing import Sequence, Optional
from pathlib import Path

import pandas as pd
import torch

from infact.common.label import Label
from infact.common.modeling import model_specifier_to_shorthand, AVAILABLE_MODELS, Model, make_model
from infact.common.logger import Logger
from infact.fact_checker import FactChecker
from infact.tools import initialize_tools
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
    """
    Simplified claim evaluation function that processes claims sequentially using ArXiv KB.
    
    Args:
        claims: List of claim texts to evaluate
        llm: Language model identifier
        tools_config: Configuration for search tools
        fact_checker_kwargs: Additional arguments for fact checker
        llm_kwargs: Additional arguments for language model
        print_log_level: Logging verbosity
        device: Optional specific device to use (defaults to cuda:0 if available)
        data_base_dir: Base directory for data storage
    """
    if llm_kwargs is None:
        llm_kwargs = dict()

    # Convert model name if needed
    llm = model_specifier_to_shorthand(llm) if llm not in AVAILABLE_MODELS["Shorthand"].values else llm
    procedure_variant = None if fact_checker_kwargs is None else fact_checker_kwargs.get("procedure_variant")

    # Setup logger
    logger = Logger(benchmark_name="NONE",
                   procedure_name=procedure_variant,
                   model_name=llm,
                   print_log_level=print_log_level,
                   target_dir="NONE")

    # Setup device
    if device is None:
        device = f"cuda:0" if torch.cuda.is_available() else "cpu"

    # Initialize ArXiv knowledge base
    arxiv_kb = ArxivKnowledgeBase(
        logger=logger,
        data_base_dir=data_base_dir,
        device=device
    )

    # Update tools config to use ArXiv KB
    if 'searcher' in tools_config:
        tools_config['searcher']['search_engine_config'] = {
            'arxiv_kb': arxiv_kb
        }

    # Initialize tools and models
    tools = initialize_tools(tools_config, logger=logger, device=device)
    llm_model = make_model(llm, logger=logger, device=device, **llm_kwargs)

    # Setup fact checker
    fc = FactChecker(
        llm=llm_model,
        tools=tools,
        logger=logger,
        **fact_checker_kwargs
    )

    results = []
    all_instance_stats = pd.DataFrame()
    start_time = time.time()

    # Process each claim
    for idx, claim_text in enumerate(claims):
        logger.set_current_fc_id(idx)
        _, docs, metas = fc.check_content(claim_text)
        doc = docs[0]
        meta = metas[0]

        # Handle statistics
        instance_stats = flatten_dict(meta["Statistics"])
        instance_stats["ID"] = idx
        instance_stats = pd.DataFrame([instance_stats])
        all_instance_stats = pd.concat([all_instance_stats, instance_stats], ignore_index=True)

        results.append({
            "id": idx,
            "claim": claim_text,
            "verdict": doc.verdict,
            "justification": doc.justification
        })

        logger.save_fc_doc(doc, idx)

    duration = time.time() - start_time
    
    stats = {
        "Total run duration": duration,
        "Time per claim": all_instance_stats["Duration"].mean(),
        "Model stats": flatten_dict(all_instance_stats.filter(like="Model").mean().to_dict()),
        "Tools stats": flatten_dict(all_instance_stats.filter(like="Tools").mean().to_dict())
    }

    return results, stats

