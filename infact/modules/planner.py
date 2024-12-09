import re
from typing import Optional, Collection

import pyparsing as pp

from infact.common.action import Action, ACTION_REGISTRY
from infact.common.document import FCDocument
from infact.common.logger import Logger
from infact.common.modeling import Model
from infact.prompts.prompt import PlanPrompt
from infact.utils.parsing import extract_last_code_block, remove_code_blocks


class Planner:
    """Chooses the next actions to perform based on the current knowledge as contained
    in the FC document."""

    def __init__(self,
                 valid_actions: Collection[type[Action]],
                 llm: Model,
                 logger: Logger,
                 extra_rules: str,
                 fall_back: str):
        assert len(valid_actions) > 0
        self.valid_actions = valid_actions
        self.llm = llm
        self.logger = logger
        self.max_tries = 5
        self.extra_rules = extra_rules
        self.fallback_action = fall_back

    def plan_next_actions(self, doc: FCDocument) -> (list[Action], str):
        performed_actions = doc.get_all_actions()
        new_valid_actions = []

        # Check if actions have been performed before adding them to valid actions
        for action_class in self.valid_actions:
            is_performed = False
            for action in performed_actions:
                if isinstance(action, action_class):
                    is_performed = True
                    break

            if not action_class.is_multimodal or (action_class.is_multimodal and not is_performed):
                new_valid_actions.append(action_class)
            else:
                self.logger.log(f"INFO: Dropping action '{action_class.name}' as it was already performed.")

        self.valid_actions = new_valid_actions
        prompt = PlanPrompt(doc, self.valid_actions, self.extra_rules)
        n_tries = 0

        while True:
            n_tries += 1
            answer = self.llm.generate(prompt)
            actions = self._extract_actions(answer)
            reasoning = self._extract_reasoning(answer)

            # Filter out actions that have been performed before
            actions = [action for action in actions if action not in performed_actions]

            if len(actions) > 0 or n_tries == self.max_tries:
                return actions, reasoning

            self.logger.log("WARNING: No new actions were found. Retrying...")

    def _extract_actions(self, answer: str) -> list[Action]:
        actions_str = extract_last_code_block(answer).replace("markdown", "")
        if not actions_str:
            candidates = []
            for action in ACTION_REGISTRY:
                pattern = re.compile(f'{action.name}("(.*?)")', re.DOTALL)
                candidates += pattern.findall(answer)
            actions_str = "\n".join(candidates)
        if not actions_str:
            # Potentially prompt LLM to correct format: Exprected format: action_name("query")
            return []
        raw_actions = actions_str.split('\n')
        actions = []
        for raw_action in raw_actions:
            action = self._parse_single_action(raw_action)
            if action:
                actions.append(action)
        return actions

    def _extract_reasoning(self, answer: str) -> str:
        return remove_code_blocks(answer).strip()

    def _parse_single_action(self, raw_action: str) -> Optional[Action]:
        if not raw_action:
            return None
        elif raw_action[0] == '"':
            raw_action = raw_action[1:]

        try:
            # Use regular expression to match action and argument in the form action(argument)
            match = re.match(r'(\w+)\((.*)\)', raw_action)

            # Extract action name and arguments
            if match:
                action_name, arguments = match.groups()
                arguments = arguments.strip()
            else:
                self.logger.log(f"Invalid action format: {raw_action}")
                match = re.search(r'"(.*?)"', raw_action)
                arguments = f'"{match.group(1)}"' if match else f'"{raw_action}"'
                first_part = raw_action.split(' ')[0]
                action_name = re.sub(r'[^a-zA-Z0-9_]', '', first_part)

            for action in ACTION_REGISTRY:
                if action_name == action.name:
                    return action(arguments)

            raise ValueError(f'Invalid action format: {raw_action} . Expected format: action_name("query")')

        except Exception as e:
            self.logger.log(f"WARNING: Failed to parse '{raw_action}':\n{e}")

        return None


def _process_answer(answer: str) -> str:
    reasoning = answer.split("NEXT_ACTIONS:")[0].strip()
    return reasoning.replace("REASONING:", "").strip()


def _extract_arguments(arguments_str: str) -> list[str]:
    """Separates the arguments_str at all commas that are not enclosed by quotes."""
    ppc = pp.pyparsing_common

    # Setup parser which separates at each comma not enclosed by a quote
    csl = ppc.comma_separated_list()

    # Parse the string using the created parser
    parsed = csl.parse_string(arguments_str)

    # Remove whitespaces and split into arguments list
    return [str.strip(value) for value in parsed]
