from abc import ABC
from typing import Optional

from infact.common.action import WebSearch
from infact.common import FCDocument, SearchResult
from infact.procedure.procedure import Procedure
from infact.prompts.prompt import AnswerQuestion
from infact.prompts.prompt import PoseQuestionsPrompt
from infact.prompts.prompt import ProposeQueries, ProposeQueriesNoQuestions
from infact.prompts.prompt import SelectBestResult
from infact.utils.console import light_blue
from infact.utils.parsing import extract_last_paragraph, find_code_span, strip_string

from rich import print

class QABased(Procedure, ABC):
    """Base class for all procedures that apply a questions & answers (Q&A) strategy."""

    def _pose_questions(self, no_of_questions: int, doc: FCDocument) -> list[str]:
        """Generates some questions that needs to be answered during the fact-check."""
        prompt = PoseQuestionsPrompt(doc, n_questions=no_of_questions)
        response = self.llm.generate(prompt)
        # Extract the questions
        questions = find_code_span(response)
        return questions

    def approach_question_batch(self, questions: list[str], doc: FCDocument) -> list:
        """Tries to answer the given list of questions. Unanswerable questions are dropped."""
        # Answer each question, one after another
        q_and_a = []
        for question in questions:
            qa_instance = self.approach_question_summarize(question, doc)
            if qa_instance is not None:
                q_and_a.append(qa_instance)

        # Add Q&A to doc reasoning
        q_and_a_strings = [(f"### {triplet['question']}\n"
                            f"Answer: {triplet['answer']}\n\n"
                            f"Source URL: {triplet['url']}") for triplet in q_and_a]
        q_and_a_string = "## Initial Q&A\n" + "\n\n".join(q_and_a_strings)
        doc.add_reasoning(q_and_a_string)

        return q_and_a

    def propose_queries_for_question(self, question: str, doc: FCDocument) -> list[WebSearch]:
        prompt = ProposeQueries(question, doc)

        n_tries = 0
        while True:
            n_tries += 1
            response = self.llm.generate(prompt)
            queries = extract_queries(response)

            if len(queries) > 0 or n_tries == self.max_attempts:
                return queries

            self.logger.log("WARNING: No new actions were found. Retrying...")

    def approach_question(self, question: str, doc: FCDocument = None) -> Optional[dict]:
        """Tries to answer the given question. If unanswerable, returns None."""
        self.logger.log(light_blue(f"Answering question: {question}"))
        self.actor.reset()
        queries = self.propose_queries_for_question(question, doc)
        if len(queries) == 0:
            return None
        search_results = self.retrieve_resources(queries)
        if len(search_results) > 0:
            answer = self.generate_answer(question, search_results, doc)
            return answer

    def answer_question(self,
                        question: str,
                        results: list[SearchResult],
                        doc: FCDocument = None) -> (str, SearchResult):
        """Answers the given question and returns the answer along with the ID of the most relevant result."""
        answer, relevant_result = self.answer_question_individually(question, results, doc)
        return answer, relevant_result

    def generate_answer(self, question: str, results: list[SearchResult], doc: FCDocument) -> Optional[dict]:
        answer, relevant_result = self.answer_question(question, results, doc)

        if answer is not None:
            self.logger.log(f"Got answer: {answer}")
            qa_instance = {"question": question,
                           "answer": answer,
                           "url": relevant_result.source,
                           "scraped_text": relevant_result.text}
            return qa_instance
        else:
            self.logger.log("Got no answer.")

    def answer_question_individually(
            self,
            question: str,
            results: list[SearchResult],
            doc: FCDocument
    ) -> (Optional[str], Optional[SearchResult]):
        """Generates an answer to the given question by iterating over the search results
        and using them individually to answer the question."""
        for result in results:
            answer = self.attempt_answer_question(question, result, doc)
            if answer is not None:
                return answer, result
        return None, None

    def attempt_answer_question(self, question: str, result: SearchResult, doc: FCDocument) -> Optional[str]:
        """Generates an answer to the given question."""
        prompt = AnswerQuestion(question, result, doc)
        response = self.llm.generate(prompt, max_attempts=3)
        # Extract answer from response
        if "NONE" not in response and "None" not in response:
            try:
                answer = extract_last_paragraph(response)
                return answer
            except:
                pass

    def approach_question_summarize(self, question: str, doc: FCDocument = None) -> Optional[dict]:
        """Tries to answer the given question using a summary-first approach. If unanswerable, returns None."""
        self.logger.log(light_blue(f"Answering question (summary approach): {question}"))
        self.actor.reset()
        queries = self.propose_queries_for_question(question, doc)
        if len(queries) == 0:
            return None
        
        # Get initial search results
        search_results = self.retrieve_resources(queries)
        if len(search_results) == 0:
            return None
        
        # Select the most promising result using summaries
        selected_result = self.select_best_result(question, search_results[:10], doc)
        if selected_result is None:
            return None
        
        # Generate answer using only the selected result
        answer = self.attempt_answer_question(question, selected_result, doc)
        if answer is not None:
            return {
                "question": question,
                "answer": answer,
                "url": selected_result.source,
                "scraped_text": selected_result.text
            }
        return None

    def select_best_result(self, question: str, results: list[SearchResult], doc: FCDocument) -> Optional[SearchResult]:
        """Selects the most promising result based on summaries."""
        summaries = [
            f"Result {i+1}:\n"
            f"Title/Source: {r.source}\n"
            f"Date: {r.date}\n"
            f"Summary: {r.summary}\n"
            for i, r in enumerate(results)
        ]
        
        prompt = SelectBestResult(question, summaries, doc)  # You'll need to create this prompt class
        response = self.llm.generate(prompt)
        
        try:
            # Assuming the LLM responds with the result number (1-10)
            selected_index = int(response.strip()) - 1
            if 0 <= selected_index < len(results):
                return results[selected_index]
        except:
            self.logger.log("WARNING: Could not parse selected result index")
        return None

def extract_queries(response: str) -> list[WebSearch]:
    matches = find_code_span(response)
    actions = []
    for match in matches:
        query = strip_string(match)
        action = WebSearch(f'"{query}"')
        actions.append(action)
    return actions
