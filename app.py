import sys
import os
import json
import logging

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn

from agent_search import search_company_options
from crawler import crawl_company
from extractor import extract_sales_brief
from config import load_config

# Suppress noisy crawl4ai logs in the web server context
logging.getLogger("crawl4ai").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

app = FastAPI(title="Sales Intelligence Pipeline")


@app.get("/", response_class=HTMLResponse)
async def index():
    template_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    with open(template_path, encoding="utf-8") as f:
        return HTMLResponse(f.read())


class SearchRequest(BaseModel):
    query: str


@app.post("/api/search")
async def search(body: SearchRequest):
    try:
        options = search_company_options(body.query)
        return {"options": options}
    except Exception as e:
        return {"error": str(e), "options": []}


@app.get("/api/brief")
async def brief_stream(url: str):
    async def generate():
        try:
            yield f"data: {json.dumps({'type': 'progress', 'step': 'crawl', 'message': 'Starting web crawl…'})}\n\n"

            markdown, crawled_urls = await crawl_company(url)

            yield f"data: {json.dumps({'type': 'progress', 'step': 'crawl_done', 'message': f'Crawled {len(crawled_urls)} pages successfully.'})}\n\n"
            yield f"data: {json.dumps({'type': 'progress', 'step': 'extract', 'message': 'Analysing content with Azure AI…'})}\n\n"

            config = load_config()
            brief = extract_sales_brief(markdown, config)

            yield f"data: {json.dumps({'type': 'complete', 'brief': brief.model_dump()})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
