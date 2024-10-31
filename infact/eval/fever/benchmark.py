import os
from pathlib import Path
from typing import Iterator
import random

import orjsonl

from config.globals import data_base_dir, random_seed
from infact.common import Label, Content
from infact.tools import WikiDumpLookup
from infact.eval.benchmark import Benchmark


class FEVER(Benchmark):
    shorthand = "fever"

    is_multimodal = False

    class_mapping = {
        "supports": Label.SUPPORTED,
        "not enough info": Label.NEI,
        "refutes": Label.REFUTED,
    }

    class_definitions = {
        Label.SUPPORTED:
            """The knowledge from the fact-check explicitly supports the entire Claim.
            That is, there is at least (!) one source that clearly (directly or
            indirectly) implies the Claim. Mere plausibility or the absence of
            opposing evidence is not enough for this decision.""",
        Label.REFUTED:
            """The knowledge from the fact-check explicitly refutes (at leas a part of) the Claim.
            That is, there is at least (!) one source that clearly (directly or
            indirectly) opposes the Claim. Mere plausibility or the absence of
            supporting evidence is not enough for this decision.""",
        Label.NEI:
            """Pick this decision if the Claim is neither `supported` nor `refuted`. For
            example, pick this decision if there is still evidence needed to clearly verify
            or refute the Claim. Before picking this decision, state which information exactly
            is missing."""
    }

    extra_prepare_rules_v1 = """* **Identify the altered segments**: Since the Claim is generated by altering
    sentences from Wikipedia, pinpoint the parts of the Claim that seem modified or out of place.
    * **Consider potential misleading elements**: The Claim may contain misleading or confusing elements due to
    the alteration process.
    * **Prepare to investigate original context**: Since the Claim is derived from Wikipedia sentences, be prepared
    to trace back to the original context or related information for accurate verification."""

    extra_plan_rules_v1 = """* **Assume the Claim may be misleading**: Since the Claim is generated by altering
    Wikipedia sentences, consider that it might be intentionally misleading or designed to test the fact-checking
    process.
    * **Identify key elements**: Break down the Claim into its key components and identify which parts require
    verification.
    * **Plan for multiple investigation steps**: The Claim may require a series of verification steps, including
    checking the original Wikipedia context, cross-referencing related information, and verifying altered segments.
    * **Consider alternative interpretations**: Given the altered nature of the Claim, consider multiple
    interpretations and verify each to ensure thorough fact-checking.
    * **Reuse previously retrieved knowledge**: Be prepared to reuse information and evidence gathered during
    previous verification steps to form a comprehensive judgment."""

    extra_prepare_rules_v2 = """* Before you start, begin with a _grammar check_ of the Claim. If it
    has some grammatical errors, there is a high chance that the Claim means something different
    than understandable at first glance. Take grammatical errors serious and elaborate on them.
    * **Take the Claim literally**: Assume that each word of the Claim is as intended. Be strict
    with the interpretation of the Claim.
    * The Claim stems from a fact-checking challenge. A human fabricated the Claim artificially 
    by using Wikipedia. The Claim could be a misleading prank, just like a trick question. It may also require
    a chain of multiple investigation steps, re-using previously retrieved knowledge."""

    extra_plan_rules_v2 = """* The Claim stems from a fact-checking challenge. A human engineered the Claim
    artificially by using Wikipedia. The Claim could be misleading, just like a trick question. It may
    also require a chain of multiple investigation steps, re-using previously retrieved knowledge."""

    available_actions = [WikiDumpLookup]

    def __init__(self, version=1, variant="dev", n_samples: int = None):
        super().__init__(f"FEVER V{version} ({variant})", variant)
        self.file_path = Path(data_base_dir + f"FEVER/fever{version}_{variant}.jsonl")
        self.justifications_file_path = Path(data_base_dir + f"FEVER/gt_justification_fever{version}_{variant}.jsonl")

        self.data = self.load_data(variant, n_samples)

        if version == 1:
            self.extra_prepare_rules = self.extra_prepare_rules_v1
            self.extra_plan_rules = self.extra_plan_rules_v1
        elif version == 2:
            self.extra_prepare_rules = self.extra_prepare_rules_v2
            self.extra_plan_rules = self.extra_plan_rules_v2
        else:
            raise ValueError(f"Invalid FEVER version '{version}' specified.")

    def load_data(self, variant, n_samples: int=None) -> list[dict]:
        # Read the files
        raw_data = orjsonl.load(self.file_path)
        justifications = None
        if n_samples:
            random.seed(random_seed)
            indices = random.sample(raw_data, len(raw_data))[:n_samples]
            raw_data = [raw_data[i] for i in indices]
            if os.path.exists(self.justifications_file_path):
                justifications_raw = orjsonl.load(self.justifications_file_path)
                justifications = [justifications_raw[i] for i in indices]

        else:
            if os.path.exists(self.justifications_file_path):
                justifications = orjsonl.load(self.justifications_file_path)

        # Translate raw data into structured list of dicts
        data = []
        for i, row in enumerate(raw_data):
            content = Content(row["claim"])
            label = self.class_mapping[row["label"].lower()] if variant in ["train", "dev"] else None
            justification = justifications[i] if justifications is not None else None
            id = row["id"]
            data.append({"id": id,
                         "content": content,
                         "label": label,
                         "justification": justification})

        return data

    def __iter__(self) -> Iterator[dict]:
        return iter(self.data)
