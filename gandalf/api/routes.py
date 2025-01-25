"""
API routes and handlers.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
from flask import Blueprint, request, jsonify, send_file, render_template
from gandalf.core.crawler import SitemapCrawler, process_urls
from gandalf.core.taxonomy import find_matching_categories
from gandalf.config.settings import get_config

# Create blueprint
api = Blueprint('api', __name__)

def format_response(data: Dict[str, Any], status: int = 200) -> tuple:
    """
    Format API response with consistent structure.
    
    Args:
        data: Response data
        status: HTTP status code
        
    Returns:
        Tuple of (response, status code)
    """
    response = {
        "status": "success" if status < 400 else "error",
        "data": data,
        "timestamp": datetime.now().isoformat()
    }
    return jsonify(response), status

@api.route('/')
async def index():
    """Render the main application page."""
    return render_template('index.html')

@api.route('/how-it-works')
async def how_it_works():
    """Render the how it works page."""
    return render_template('how_it_works.html')

@api.route('/fetch_sitemap_info', methods=['POST'])
async def get_sitemap_info():
    """
    Fetch sitemap information for a given URL.
    
    Returns:
        JSON response with sitemap information
    """
    url = request.form.get('url')
    if not url:
        return format_response({"error": "URL is required"}, 400)
    
    # Normalize URL
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    try:
        async with SitemapCrawler() as crawler:
            urls, sitemap_info = await crawler.fetch_sitemap(url)
            
            return format_response({
                "sitemap_info": sitemap_info,
                "total_urls_found": len(urls),
                "urls": urls
            })
    except Exception as e:
        return format_response({"error": str(e)}, 500)

@api.route('/crawl', methods=['POST'])
async def crawl():
    """
    Process URLs and categorize their content.
    
    Returns:
        JSON response with crawl results
    """
    config = get_config()
    data = request.get_json()
    
    if not data or 'urls' not in data or 'limit' not in data:
        return format_response({"error": "URLs and limit are required"}, 400)
    
    urls = data['urls']
    limit = int(data['limit'])
    total_urls_found = len(urls)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    try:
        # Process URLs
        results = await process_urls(urls, limit)
        
        # Add category matches to results
        for result in results:
            if "error" not in result and "content" in result:
                matches = find_matching_categories(result["content"])
                result["categories"] = [{
                    "path": match.get_path_with_similarities(),
                    "similarity": round(match.max_similarity, 2)
                } for match in matches]
                del result["content"]  # Remove content to reduce response size
        
        # Prepare output
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
                "crawl_time": datetime.now().isoformat(),
                "detailed_reports": {
                    "csv": f"crawl_details_{timestamp}.csv",
                    "json": f"crawl_details_{timestamp}.json"
                }
            },
            "results": results
        }
        
        # Save results to file
        filename = config.RESULTS_DIR / f"crawl_results_{timestamp}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        return format_response(output)
    except Exception as e:
        return format_response({"error": str(e)}, 500)

@api.route('/download/<filename>')
async def download(filename):
    """
    Download results file.
    
    Args:
        filename: Name of the file to download
        
    Returns:
        File download response
    """
    config = get_config()
    try:
        file_path = config.RESULTS_DIR / filename
        if not file_path.exists():
            return format_response({"error": "File not found"}, 404)
            
        # Set appropriate content type
        content_type = 'text/csv' if filename.endswith('.csv') else 'application/json'
        return send_file(
            file_path,
            as_attachment=True,
            mimetype=content_type
        )
    except Exception as e:
        return format_response({"error": str(e)}, 404) 