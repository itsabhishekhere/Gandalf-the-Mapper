"""
Core crawler module for fetching and processing web content.
"""
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import aiohttp
from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
import csv
import json
from pathlib import Path
from gandalf.config.settings import get_config

# Configure logging
logger = logging.getLogger(__name__)
config = get_config()

class SitemapCrawler:
    """Handles sitemap discovery and URL extraction."""
    
    def __init__(self):
        """Initialize the sitemap crawler."""
        self.session = None
    
    async def __aenter__(self):
        """Set up async context."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up async context."""
        if self.session:
            await self.session.close()
    
    async def fetch_sitemap(self, url: str) -> Tuple[List[str], Dict[str, List[str]]]:
        """
        Fetch and parse sitemap URLs.
        
        Args:
            url: Base URL to fetch sitemap from
            
        Returns:
            Tuple of (all URLs, sitemap information)
        """
        sitemap_info = {
            "main_sitemaps": [],
            "sub_sitemaps": {}
        }
        
        try:
            # Try common sitemap locations
            sitemap_urls = [
                f"{url.rstrip('/')}/sitemap.xml",
                f"{url.rstrip('/')}/sitemap_index.xml",
                f"{url.rstrip('/')}/sitemap-index.xml"
            ]
            
            all_urls = []
            for sitemap_url in sitemap_urls:
                try:
                    async with self.session.get(sitemap_url) as response:
                        if response.status == 200:
                            content = await response.text()
                            soup = BeautifulSoup(content, 'xml')
                            sitemap_info["main_sitemaps"].append(sitemap_url)
                            
                            # Extract URLs from both sitemap index and regular sitemaps
                            urls = []
                            for loc in soup.find_all('loc'):
                                if loc.text.endswith('.xml'):
                                    # This is a sitemap index, fetch the sub-sitemap
                                    sub_urls = []
                                    async with self.session.get(loc.text) as sub_response:
                                        if sub_response.status == 200:
                                            sub_content = await sub_response.text()
                                            sub_soup = BeautifulSoup(sub_content, 'xml')
                                            sub_urls = [url.text for url in sub_soup.find_all('loc')]
                                            urls.extend(sub_urls)
                                            sitemap_info["sub_sitemaps"][loc.text] = sub_urls
                                else:
                                    urls.append(loc.text)
                            all_urls.extend(urls)
                            logger.info(f"Found {len(urls)} URLs in {sitemap_url}")
                            return all_urls, sitemap_info
                except Exception as e:
                    logger.warning(f"Error fetching sitemap {sitemap_url}: {e}")
                    continue
            
            # If no sitemap found, return the base URL
            logger.info("No sitemap found, using base URL")
            return [url], {"main_sitemaps": [], "sub_sitemaps": {}, "note": "No sitemap found, using base URL"}
        except Exception as e:
            logger.error(f"Error in fetch_sitemap: {e}")
            return [url], {"main_sitemaps": [], "sub_sitemaps": {}, "error": str(e)}

class ContentCrawler:
    """Handles content crawling and extraction."""
    
    def __init__(self):
        """Initialize the content crawler."""
        self.browser_config = BrowserConfig(headless=True)
        self.run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
        self.crawl_details = []

    async def crawl_url(self, url: str) -> Dict:
        """
        Crawl a single URL and extract content.
        
        Args:
            url: URL to crawl
            
        Returns:
            Dictionary containing crawl results
        """
        try:
            async with AsyncWebCrawler(config=self.browser_config) as crawler:
                result = await crawler.arun(url=url, config=self.run_config)
                
                if result.success:
                    # Parse HTML
                    soup = BeautifulSoup(result.html, 'html.parser')
                    title = soup.title.string if soup.title else "No title"
                    
                    # Extract meta description
                    meta_desc = soup.find('meta', attrs={'name': 'description'})
                    description = meta_desc['content'] if meta_desc else None
                    
                    # Count elements before filtering
                    pre_filter_stats = {
                        'total_elements': len(soup.find_all()),
                        'navigation_elements': len(soup.find_all(['nav', 'header', 'footer'])),
                        'script_elements': len(soup.find_all('script')),
                        'style_elements': len(soup.find_all('style')),
                        'ad_elements': len(soup.find_all(class_=lambda x: x and ('ad' in x.lower() or 'promo' in x.lower()))),
                        'sidebar_elements': len(soup.find_all(class_=lambda x: x and 'sidebar' in x.lower())),
                    }
                    
                    # Get raw text content before filtering
                    raw_text = soup.get_text(separator=' ', strip=True)
                    
                    # Use the markdown content from crawl4ai (filtered content)
                    filtered_content = result.markdown
                    
                    # Store crawl details
                    crawl_detail = {
                        'url': url,
                        'timestamp': datetime.now().isoformat(),
                        'title': title,
                        'description': description,
                        'pre_filter_stats': pre_filter_stats,
                        'raw_content_length': len(raw_text),
                        'filtered_content_length': len(filtered_content),
                        'elements_removed': pre_filter_stats['total_elements'] - len(filtered_content.split()),
                        'raw_text': raw_text[:1000],  # Store first 1000 chars of raw text
                        'filtered_text': filtered_content[:1000],  # Store first 1000 chars of filtered text
                    }
                    self.crawl_details.append(crawl_detail)
                    
                    logger.info(f"Successfully crawled {url}")
                    return {
                        "url": url,
                        "title": title,
                        "description": description,
                        "content": filtered_content,
                        "content_length": len(filtered_content),
                        "crawled_at": datetime.now().isoformat()
                    }
                else:
                    logger.warning(f"Failed to crawl {url}: {result.error_message}")
                    return {
                        "url": url,
                        "error": result.error_message,
                        "crawled_at": datetime.now().isoformat()
                    }
        except Exception as e:
            logger.error(f"Error crawling {url}: {e}")
            return {
                "url": url,
                "error": str(e),
                "crawled_at": datetime.now().isoformat()
            }

    def export_crawl_details(self, timestamp: str) -> Tuple[str, str]:
        """
        Export crawl details to CSV and JSON files.
        
        Args:
            timestamp: Timestamp string for filenames
            
        Returns:
            Tuple of (csv_path, json_path)
        """
        if not self.crawl_details:
            return None, None
            
        # Create export paths
        csv_path = config.RESULTS_DIR / f"crawl_details_{timestamp}.csv"
        json_path = config.RESULTS_DIR / f"crawl_details_{timestamp}.json"
        
        # Export to CSV
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'url', 'timestamp', 'title', 'description',
                'raw_content_length', 'filtered_content_length', 'elements_removed',
                'total_elements', 'navigation_elements', 'script_elements',
                'style_elements', 'ad_elements', 'sidebar_elements',
                'raw_text', 'filtered_text'
            ])
            writer.writeheader()
            for detail in self.crawl_details:
                row = {
                    'url': detail['url'],
                    'timestamp': detail['timestamp'],
                    'title': detail['title'],
                    'description': detail['description'],
                    'raw_content_length': detail['raw_content_length'],
                    'filtered_content_length': detail['filtered_content_length'],
                    'elements_removed': detail['elements_removed'],
                    'total_elements': detail['pre_filter_stats']['total_elements'],
                    'navigation_elements': detail['pre_filter_stats']['navigation_elements'],
                    'script_elements': detail['pre_filter_stats']['script_elements'],
                    'style_elements': detail['pre_filter_stats']['style_elements'],
                    'ad_elements': detail['pre_filter_stats']['ad_elements'],
                    'sidebar_elements': detail['pre_filter_stats']['sidebar_elements'],
                    'raw_text': detail['raw_text'],
                    'filtered_text': detail['filtered_text']
                }
                writer.writerow(row)
        
        # Export to JSON (for full details)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.crawl_details, f, indent=2, ensure_ascii=False)
            
        return str(csv_path), str(json_path)

async def process_urls(urls: List[str], limit: Optional[int] = None) -> List[Dict]:
    """
    Process a list of URLs with content crawling.
    
    Args:
        urls: List of URLs to process
        limit: Optional limit on number of URLs to process
        
    Returns:
        List of crawl results
    """
    if limit:
        urls = urls[:limit]
    
    crawler = ContentCrawler()
    results = []
    
    for url in urls:
        try:
            result = await crawler.crawl_url(url)
            results.append(result)
        except Exception as e:
            logger.error(f"Error processing {url}: {e}")
            results.append({
                "url": url,
                "error": f"Failed to process: {str(e)}",
                "crawled_at": datetime.now().isoformat()
            })
    
    # Export crawl details
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path, json_path = crawler.export_crawl_details(timestamp)
    
    if csv_path:
        logger.info(f"Exported crawl details to CSV: {csv_path}")
        logger.info(f"Exported crawl details to JSON: {json_path}")
    
    return results 