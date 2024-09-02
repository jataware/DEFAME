from typing import Optional

from infact.common import FCDocument, SearchResult, Action
from infact.procedure.variants.qa_based.infact import InFact
from infact.prompts.prompt import ProposeQuerySimple
from infact.procedure.variants.qa_based.base import extract_queries


class SimpleQA(InFact):
    """InFact but without interpretation, uses only one query per question and takes first search result.
    (Never used in AVeriTeC challenge)."""

    def propose_queries_for_question(self, question: str, doc: FCDocument) -> list[Action]:
        prompt = ProposeQuerySimple(question)

        n_tries = 0
        while n_tries < self.max_attempts:
            n_tries += 1
            response = self.llm.generate(prompt)
            queries = extract_queries(response)

            if len(queries) > 0:
                return [queries[0]]

            self.logger.log("No new actions were found. Retrying...")

        self.logger.warning("Got no search query, dropping this question.")
        return []

    def answer_question(self,
                        question: str,
                        results: list[SearchResult],
                        doc: FCDocument = None) -> (str, SearchResult):
        relevant_result = results[0]
        answer = self.attempt_answer_question(question, relevant_result, doc)
        return answer, relevant_result
