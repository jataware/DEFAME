import time
from typing import List, Dict

from duckduckgo_search import DDGS

from infact.common.misc import Query, WebSource
from infact.tools.search.remote_search_api import RemoteSearchAPI
from infact.utils.console import red, bold


class DuckDuckGo(RemoteSearchAPI):
    """Class for querying the DuckDuckGo API."""
    name = "duckduckgo"

    def __init__(self,
                 max_retries: int = 10,
                 backoff_factor: float = 60.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.total_searches = 0

    def _call_api(self, query: Query) -> list[WebSource]:
        """Run a search query and return structured results."""
        # TODO: Implement start and end date
        attempt = 0
        while attempt < self.max_retries:
            if attempt > 3:
                wait_time = self.backoff_factor * (attempt)
                self.logger.log(bold(red(f"Sleeping {wait_time} seconds.")))
                time.sleep(wait_time)
            try:
                self.total_searches += 1
                response = DDGS().text(query.text, max_results=query.limit)
                if not response:
                    self.logger.log(bold(red("DuckDuckGo is having issues. Run duckduckgo.py "
                                             "and check https://duckduckgo.com/ for more information.")))
                    return []
                return self._parse_results(response, query)
            except Exception as e:
                attempt += 1
                query += '?'
                self.logger.log((bold(red(f"Attempt {attempt} failed: {e}. Retrying with modified query..."))))
        self.logger.log(bold(red("All attempts to reach DuckDuckGo have failed. Please try again later.")))

        return []

    def _parse_results(self, response: List[Dict[str, str]], query: Query) -> list[WebSource]:
        """Parse results from DuckDuckGo search and return structured dictionary."""
        results = []
        for i, result in enumerate(response):
            url = result.get('href', '')
            title = result.get('title', '')
            text = result.get('body', '')
            
            results.append(WebSource(url=url, title=title, text=text, query=query, rank=i))
        return results
