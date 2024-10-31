import os
from pathlib import Path
from typing import Iterator
import pandas as pd

from config.globals import data_base_dir, random_seed
from infact.common import Label, Content
from infact.common.medium import Image
from infact.eval.benchmark import Benchmark
from infact.tools import WebSearch, ImageSearch, ReverseSearch

class MOCHEG(Benchmark):
    shorthand = "mocheg"
    is_multimodal = True

    class_mapping = {
        "supported": Label.SUPPORTED,
        "refuted": Label.REFUTED,
        "nei": Label.NEI,
    }

    class_definitions = {
        Label.SUPPORTED:
            "A claim is considered supported when the provided evidence backs up the claim.",
        Label.REFUTED:
            "A claim is considered refuted when the evidence contradicts the claim.",
        Label.NEI:
            "A claim is marked as NEI when there isn't enough evidence to support or refute the claim."
    }

    available_actions = [WebSearch, ImageSearch]

    def __init__(self, variant="val", n_samples: int = None):
        super().__init__(f"MOCHEG ({variant})", variant)
        self.file_path = Path(data_base_dir + f"MOCHEG/{variant}/Corpus2.csv")
        self.image_path = Path(data_base_dir + f"MOCHEG/{variant}/images/")
        self.data = self.load_data(n_samples)

    def load_data(self, n_samples: int = None) -> list[dict]:
        # Load the corpus
        df = pd.read_csv(self.file_path)
        if n_samples and n_samples < len(df):
            df = df.sample(n=n_samples, random_state=random_seed)

        data = []
        for i, row in df.iterrows():
            #image_file = f"{row['claim_id']}-proof-{row['evidence_id']}.jpg" # this is not correct yet
            #image_path = self.image_path / image_file


            # Load the image evidence
            #if os.path.exists(image_path):
            #    image = Image(image_path)
            #else:
            #    image = None  # Or handle missing images
            image = None

            claim_text = row["Claim"]
            text_evidence = row["Evidence"]
            label = self.class_mapping[row["cleaned_truthfulness"].lower()]
            id = f'{row["claim_id"]}'

            # Create an entry for each claim
            entry = {
                "id": id,
                "content": Content(text=claim_text, id_number=id),
                "label": label,
                "justification": row.get("ruling_outline", "")
            }
            data.append(entry)

        return data

    def __iter__(self) -> Iterator[dict]:
        return iter(self.data)
