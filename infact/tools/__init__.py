from infact.common.logger import Logger
from infact.tools.search.searcher import Searcher
from infact.tools.tool import Tool, get_available_actions

TOOL_REGISTRY = [
    Searcher,
]


def get_tool_by_name(name: str):
    for t in TOOL_REGISTRY:
        if t.name == name:
            return t
    raise ValueError(f'Tool with name "{name}" does not exist.')


def initialize_tools(config: dict[str, dict], logger: Logger, device=None) -> list[Tool]:
    tools = []
    for tool_name, kwargs in config.items():
        tool_class = get_tool_by_name(tool_name)
        t = tool_class(**kwargs, logger=logger, device=device)
        tools.append(t)
    return tools
