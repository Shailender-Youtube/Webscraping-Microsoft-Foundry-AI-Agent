"""
Crawl4AI — Basic Demo
=====================
This script shows how Crawl4AI crawls a webpage and converts it
into clean Markdown that is ready to be fed into an AI model.

Run:
    python crawl_test.py
"""

import sys
import asyncio

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator


URL = "https://www.bbc.com/news"   # ← change this to any website you like


async def main():
    print("=" * 60)
    print("  Crawl4AI — Basic Demo")
    print("=" * 60)
    print(f"\n  Target URL : {URL}")
    print("  Launching headless browser...\n")

    # 1. Browser config — run Chrome headlessly (invisible)
    browser_config = BrowserConfig(headless=True)

    # 2. Content filter — strips navbars, ads, footers (keeps signal only)
    #    Markdown generator — converts the clean HTML into Markdown for LLMs
    md_generator = DefaultMarkdownGenerator(
        content_filter=PruningContentFilter()
    )

    # 3. Run config — bypass cache so we always get fresh content
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        markdown_generator=md_generator,
    )

    async with AsyncWebCrawler(config=browser_config, verbose=False) as crawler:
        result = await crawler.arun(url=URL, config=run_config)

    if not result.success:
        print(f"  ❌  Crawl failed: {result.error_message}")
        return

    # 4. Get the clean Markdown output
    md = result.markdown
    content = (md.fit_markdown or md.raw_markdown) if md else ""

    if not content:
        print("  ⚠️  No content extracted.")
        return

    word_count  = len(content.split())
    char_count  = len(content)
    line_count  = content.count("\n")

    print("  ✅  Crawl successful!\n")
    print("-" * 60)
    print("  STATS")
    print("-" * 60)
    print(f"  Words      : {word_count:,}")
    print(f"  Characters : {char_count:,}")
    print(f"  Lines      : {line_count:,}")
    print("-" * 60)
    print("\n  CONTENT PREVIEW  (first 100 lines)")
    print("-" * 60)

    preview_lines = content.splitlines()[:100]
    print("\n".join(preview_lines))

    print("\n" + "-" * 60)
    print("  ... (truncated for display)")
    print("-" * 60)
    print("\n  This is the clean Markdown that gets sent to the AI model.")
    print("  No raw HTML, no noise — just structured content ready for LLMs.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
