from abc import ABC
from typing import Any

import torch

from infact.common.action import Action
from infact.common.results import Result
from infact.common.logger import Logger


class Tool(ABC):
    """Base class for all tools."""
    name: str
    actions: list[type(Action)]  # (classes of the) available actions this tool offers

    def __init__(self, logger: Logger = None, device: str | torch.device = None):
        self.logger = logger or Logger()
        self.device = device

    def perform(self, action: Action) -> list[Result]:
        raise NotImplementedError

    def reset(self) -> None:
        """Resets the tool to its initial state (if applicable) and sets all stats to zero."""
        pass

    def get_stats(self) -> dict[str, Any]:
        """Returns the tool's usage statistics as a dictionary."""
        return {}


def get_available_actions(tools: list[Tool]) -> set[type[Action]]:
    actions = set()
    for tool in tools:
        actions.update(tool.actions)
    return actions