import os
import asyncio
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from bs4 import BeautifulSoup
import aiohttp
from typing import List, Dict, Tuple
from content_taxonomy import find_matching_categories, CategoryMatch

app = Flask(__name__)

async def fetch_sitemap(url: str) -> Tuple[List[str], Dict[str, List[str]]]:
    """Fetch and parse sitemap URLs."""
    sitemap_info = {
        "main_sitemaps": [],
        "sub_sitemaps": {}
    }
    try:
        async with aiohttp.ClientSession() as session:
            # Try common sitemap locations
            sitemap_urls = [
                f"{url.rstrip('/')}/sitemap.xml",
                f"{url.rstrip('/')}/sitemap_index.xml",
                f"{url.rstrip('/')}/sitemap-index.xml"
            ]
            
            all_urls = []
            for sitemap_url in sitemap_urls:
                try:
                    async with session.get(sitemap_url) as response:
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
                                    async with session.get(loc.text) as sub_response:
                                        if sub_response.status == 200:
                                            sub_content = await sub_response.text()
                                            sub_soup = BeautifulSoup(sub_content, 'xml')
                                            sub_urls = [url.text for url in sub_soup.find_all('loc')]
                                            urls.extend(sub_urls)
                                            sitemap_info["sub_sitemaps"][loc.text] = sub_urls
                                else:
                                    urls.append(loc.text)
                            all_urls.extend(urls)
                            return all_urls, sitemap_info
                except:
                    continue
            
            # If no sitemap found, return the base URL
            return [url], {"main_sitemaps": [], "sub_sitemaps": {}, "note": "No sitemap found, using base URL"}
    except Exception as e:
        print(f"Error fetching sitemap: {e}")
        return [url], {"main_sitemaps": [], "sub_sitemaps": {}, "error": str(e)}

async def crawl_url(url: str) -> Dict:
    """Crawl a single URL and extract content."""
    try:
        browser_config = BrowserConfig(headless=True)
        run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
        
        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await crawler.arun(url=url, config=run_config)
            
            if result.success:
                # Extract title and content
                soup = BeautifulSoup(result.html, 'html.parser')
                title = soup.title.string if soup.title else "No title"
                
                # Use the markdown content from crawl4ai
                content = result.markdown
                
                # Find matching categories from taxonomy
                matches = find_matching_categories(content)
                
                # Convert matches to dictionary format with paths
                categories = [{
                    "path": match.get_path_with_similarities(),
                    "similarity": round(match.max_similarity, 2)
                } for match in matches]
                
                # Extract meta description if available
                meta_desc = soup.find('meta', attrs={'name': 'description'})
                description = meta_desc['content'] if meta_desc else None
                
                return {
                    "url": url,
                    "title": title,
                    "description": description,
                    "content_length": len(content),
                    "categories": categories,
                    "crawled_at": datetime.now().isoformat()
                }
            else:
                return {
                    "url": url,
                    "error": result.error_message,
                    "crawled_at": datetime.now().isoformat()
                }
    except Exception as e:
        return {
            "url": url,
            "error": str(e),
            "crawled_at": datetime.now().isoformat()
        }

@app.route('/')
async def index():
    return render_template('index.html')

@app.route('/fetch_sitemap_info', methods=['POST'])
async def get_sitemap_info():
    url = request.form.get('url')
    if not url:
        return jsonify({"error": "URL is required"}), 400
    
    # Normalize URL
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    try:
        # Fetch sitemap URLs
        urls, sitemap_info = await fetch_sitemap(url)
        total_urls_found = len(urls)
        
        return jsonify({
            "sitemap_info": sitemap_info,
            "total_urls_found": total_urls_found,
            "urls": urls  # We'll need these URLs for the next step
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/crawl', methods=['POST'])
async def crawl():
    data = request.get_json()
    if not data or 'urls' not in data or 'limit' not in data:
        return jsonify({"error": "URLs and limit are required"}), 400
    
    urls = data['urls']
    limit = int(data['limit'])
    total_urls_found = len(urls)
    
    try:
        # Limit URLs based on user input
        urls = urls[:limit]
        
        # Crawl each URL
        results = []
        for url in urls:
            try:
                result = await crawl_url(url)
                results.append(result)
            except Exception as e:
                results.append({
                    "url": url,
                    "error": f"Failed to crawl: {str(e)}",
                    "crawled_at": datetime.now().isoformat()
                })
        
        # Save results to a file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"crawl_results_{timestamp}.txt"
        
        output = {
            "sitemap_info": {
                "total_urls_available": total_urls_found,
                "urls_selected": limit
            },
            "crawl_summary": {
                "total_urls_found": total_urls_found,
                "urls_selected": limit,
                "urls_crawled": len(results),
                "successful_crawls": sum(1 for r in results if "error" not in r),
                "failed_crawls": sum(1 for r in results if "error" in r),
                "mapped_urls": sum(1 for r in results if "error" not in r and r.get("categories") and len(r["categories"]) > 0),
                "unmapped_urls": sum(1 for r in results if "error" not in r and (not r.get("categories") or len(r["categories"]) == 0)),
                "crawl_time": datetime.now().isoformat()
            },
            "results": results
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        return jsonify(output)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/download/<filename>')
async def download(filename):
    try:
        return send_file(filename, as_attachment=True)
    except Exception as e:
        return jsonify({"error": str(e)}), 404

if __name__ == '__main__':
    app.run(debug=True) 