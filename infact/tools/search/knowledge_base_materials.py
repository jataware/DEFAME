"""
This file contains the MaterialsKnowledgeBase class, which performs semantic search
through the MDPI Materials journal API.
"""

import requests
from bs4 import BeautifulSoup
from datetime import datetime
from typing import List, Optional
from time import time
from infact.tools.search.local_search_api import LocalSearchAPI
from infact.tools.search.knowledge_base import SearchResult
from rich import print

class MaterialsKnowledgeBase(LocalSearchAPI):
    """The MDPI Materials Knowledge Base used to retrieve relevant papers."""
    name = 'materials_kb'
    is_free = True

    def __init__(
        self,
        variant: str = "materials",
        logger = None,
    ):
        super().__init__(logger=logger)
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def _get_full_text(self, article_url: str) -> Optional[str]:
        """Get full text content of an article."""
        response = requests.get(article_url, headers=self.headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        article_div = soup.find('div', class_='html-body')
        
        if not article_div:
            return None

        # Process all elements
        formatted_text = []
        current_section = None
        
        for element in article_div.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'div']):
            # Skip certain div classes
            if element.name == 'div' and element.get('class') and any(c in ['figure', 'table', 'references'] for c in element.get('class')):
                continue
                
            text = element.get_text().strip()
            if not text:
                continue
                
            # If it's a header, update the current section
            if element.name.startswith('h'):
                current_section = text
                formatted_text.append(f"\n## {text}\n")
            # If it's content, add it under the current section
            elif current_section and (element.name == 'p' or element.name == 'div'):
                formatted_text.append(text)

        return "\n".join(formatted_text)

    def _search_materials(self, query: str, page: int = 1) -> List[dict]:
        """Search MDPI Materials journal."""
        base_url = "https://www.mdpi.com/search"
        params = {
            'q': query,
            'journal': 'materials',
            'page_no': page,
            'article_type': 'research-article',
            'search_type': 'academic'
        }
        try:
            response = requests.get(base_url, params=params, headers=self.headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            articles = soup.find_all('div', class_='article-content')
            results = []
            for article in articles:
                title_elem = article.find('a', class_='title-link')
                abstract_elem = article.find('div', class_='abstract-full')                
                
                if title_elem:
                    article_data = {
                        'title': title_elem.text.strip(),
                        'url': 'https://www.mdpi.com' + title_elem['href'],
                        'abstract': abstract_elem.text.strip() if abstract_elem else None,
                    }
                    results.append(article_data)
                                
            return results
        except requests.RequestException as e:
            self.logger.error(f"Error fetching results: {e}")
            return []

    def _call_api(self, query: str, limit: int) -> List[SearchResult]:
        """Search the MDPI Materials journal and return results."""
        start_time = time()
        
        # Get initial search results
        results = []
        page = 1
        while len(results) < limit:
            articles = self._search_materials(query, page)
            if not articles:
                break
                
            for article in articles:
                if len(results) >= limit:
                    break
                    
                # Get full text for better context
                full_text = self._get_full_text(article['url'])
                text = full_text if full_text else article['abstract']
                
                results.append(SearchResult(
                    source=article['url'],
                    text=text,
                    query=query,
                    rank=len(results),
                    date=None
                ))
                    
        return results

    def _before_search(self, query: str) -> str:
        """Pre-process the search query if needed."""
        return query 