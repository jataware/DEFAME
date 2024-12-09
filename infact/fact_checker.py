import time
from typing import Sequence, Any

import numpy as np

from infact.common.claim import Claim
from infact.common.content import Content
from infact.common.document import FCDocument
from infact.common.label import Label
from infact.common.modeling import Model, make_model
from infact.modules.actor import Actor
from infact.modules.doc_summarizer import DocSummarizer
from infact.modules.judge import Judge
from infact.modules.planner import Planner
from infact.procedure import get_procedure
from infact.tools import *
from infact.utils.console import gray, light_blue, bold, sec2mmss


class FactChecker:
    """The core class for end-to-end fact verification."""

    def __init__(self,
                 llm: str | Model = "gpt_4o_mini",
                 tools: list[Tool] = None,
                 search_engines: dict[str, dict] = None,
                 procedure_variant: str = "infact",
                 max_iterations: int = 5,
                 max_result_len: int = None,
                 logger: Logger = None,
                 classes: Sequence[Label] = None,
                 class_definitions: dict[Label, str] = None,
                 extra_prepare_rules: str = None,
                 extra_plan_rules: str = None,
                 extra_judge_rules: str = None,
                 print_log_level: str = "info",
                 ):
        assert not tools or not search_engines, \
            "You are allowed to specify either tools or search engines."

        self.logger = logger or Logger(print_log_level=print_log_level)

        self.llm = make_model(llm, logger=self.logger) if isinstance(llm, str) else llm

        if classes is None:
            if class_definitions is None:
                classes = [Label.SUPPORTED, Label.NEI, Label.REFUTED]
            else:
                classes = list(class_definitions.keys())

        self.extra_prepare_rules = extra_prepare_rules
        self.max_iterations = max_iterations
        self.max_result_len = max_result_len

        if tools is None:
            tools = self._initialize_tools(search_engines)
        self.fall_back_action = tools[0].actions[0]
        self.logger.log(f"Selecting {self.fall_back_action.name} as fallback option if no action can be matched.")

        available_actions = get_available_actions(tools)

        # Initialize fact-checker modules
        self.planner = Planner(valid_actions=available_actions,
                               llm=self.llm,
                               logger=self.logger,
                               extra_rules=extra_plan_rules,
                               fall_back=self.fall_back_action)

        self.actor = Actor(tools=tools, llm=self.llm, logger=self.logger)

        self.judge = Judge(llm=self.llm,
                           logger=self.logger,
                           classes=classes,
                           class_definitions=class_definitions,
                           extra_rules=extra_judge_rules)

        self.doc_summarizer = DocSummarizer(self.llm, self.logger)

        self.procedure = get_procedure(procedure_variant,
                                       llm=self.llm,
                                       actor=self.actor,
                                       judge=self.judge,
                                       planner=self.planner,
                                       logger=self.logger,
                                       max_iterations=self.max_iterations)

    def _initialize_tools(self, search_engines: dict[str, dict]) -> list[Tool]:
        """Loads a default collection of tools."""
        # Unimodal tools
        tools = [
            Searcher(search_engines, max_result_len=self.max_result_len, logger=self.logger),
        ]
        return tools

    def check_content(self, content: Content | str) -> (Label, list[FCDocument], list[dict[str, Any]]):
        """
        Fact-checks the given content ent-to-end by first extracting all check-worthy claims and then
        verifying each claim individually. Returns the aggregated veracity and the list of corresponding
        fact-checking documents, one doc per claim.
        """
        start = time.time()

        content = Content(content) if isinstance(content, str) else content

        self.logger.info(bold(f"Content to be checked:\n'{light_blue(str(content))}'"))

        claims = [Claim(content.text, content)]

        # Verify each single extracted claim
        self.logger.log(bold("Verifying the claims..."))
        docs = []
        metas = []
        for claim in claims:
            doc, meta = self.verify_claim(claim)
            docs.append(doc)
            metas.append(meta)

        aggregated_veracity = aggregate_predictions([doc.verdict for doc in docs])
        self.logger.log(bold(f"So, the overall veracity is: {aggregated_veracity.value}"))
        fc_duration = time.time() - start
        self.logger.log(f"Fact-check took {sec2mmss(fc_duration)}.")
        return aggregated_veracity, docs, metas

    def verify_claim(self, claim: Claim) -> (FCDocument, dict[str, Any]):
        """Takes an (atomic, decontextualized, check-worthy) claim and fact-checks it.
        This is the core of the fact-checking implementation. Here, the fact-checking
        document is constructed incrementally."""
        stats = {}
        self.actor.reset()  # remove all past search evidences
        self.llm.reset_stats()

        start = time.time()
        doc = FCDocument(claim)

        # Depending on the specified procedure variant, perform the fact-check
        label, meta = self.procedure.apply_to(doc)

        # Finalize the fact-check
        doc.add_reasoning("## Final Judgement\n" + self.judge.get_latest_reasoning())

        # Summarize the fact-check and use the summary as justification
        if label == Label.REFUSED_TO_ANSWER:
            self.logger.warning("The model refused to answer.")
        else:
            doc.justification = self.doc_summarizer.summarize(doc)
            self.logger.info(bold(f"The claim '{light_blue(str(claim.text))}' is {label.value}."))
            self.logger.info(f'Justification: {gray(doc.justification)}')
        doc.verdict = label

        stats["Duration"] = time.time() - start
        stats["Model"] = self.llm.get_stats()
        stats["Tools"] = self.actor.get_tool_stats()
        meta["Statistics"] = stats
        return doc, meta


def aggregate_predictions(veracities: Sequence[Label]) -> Label:
    veracities = np.array(veracities)
    if np.any(veracities == Label.REFUSED_TO_ANSWER):
        return Label.REFUSED_TO_ANSWER
    elif np.all(veracities == Label.SUPPORTED):
        return Label.SUPPORTED
    elif np.any(veracities == Label.REFUTED):
        return Label.REFUTED
    elif np.any(veracities == Label.CONFLICTING):
        return Label.CONFLICTING
    elif np.any(veracities == Label.CHERRY_PICKING):
        return Label.CHERRY_PICKING
    else:
        return Label.NEI
