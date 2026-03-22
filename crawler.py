import logging
from urllib.parse import urlparse, urljoin
from typing import Tuple, List

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

logger = logging.getLogger(__name__)

TARGET_PATHS = [
    "/",
    "/about",
    "/about-us",
    "/blog",
    "/news",
    "/newsroom",
    "/press",
    "/press-releases",
    "/careers",
    "/jobs",
    "/investors",
    "/investor-relations",
    "/partners",
    "/partnerships",
    "/solutions",
    "/products",
    "/services",
]


def _build_target_urls(base_url: str) -> List[str]:
    parsed = urlparse(base_url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    return [urljoin(base, path) for path in TARGET_PATHS]


def _is_target_url(url: str, base_url: str) -> bool:
    parsed_base = urlparse(base_url)
    parsed_url = urlparse(url)
    if parsed_url.netloc != parsed_base.netloc:
        return False
    path = parsed_url.path.rstrip("/") or "/"
    for target in TARGET_PATHS:
        if path == target.rstrip("/") or path.startswith(target.rstrip("/") + "/"):
            return True
    return False


async def crawl_company(url: str) -> Tuple[str, List[str]]:
    """
    Crawl a company website and return combined Markdown and list of crawled URLs.
    """
    browser_config = BrowserConfig(headless=True)

    md_generator = DefaultMarkdownGenerator(
        content_filter=PruningContentFilter()
    )

    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        markdown_generator=md_generator,
        deep_crawl_strategy=None,  # we manually crawl target URLs
    )

    target_urls = _build_target_urls(url)
    crawled_pages = []
    failed_urls = []

    async with AsyncWebCrawler(config=browser_config, verbose=False) as crawler:
        for target_url in target_urls:
            try:
                result = await crawler.arun(url=target_url, config=run_config)
                md = result.markdown if result.success and result.markdown else None
                raw = (md.fit_markdown or md.raw_markdown) if md else None
                if raw and raw.strip():
                    crawled_pages.append((target_url, raw.strip()))
                    logger.info(f"Crawled: {target_url}")
                else:
                    logger.info(f"No content or failed, skipping: {target_url}")
                    failed_urls.append(target_url)
            except Exception as e:
                logger.warning(f"Error crawling {target_url}: {e}")
                failed_urls.append(target_url)

    if len(crawled_pages) < 2:
        raise RuntimeError(
            f"Only {len(crawled_pages)} page(s) crawled successfully from {url}. "
            "Need at least 2 pages to generate a meaningful brief. "
            f"Failed URLs: {failed_urls}"
        )

    combined_markdown = "\n\n---\n\n".join(content for _, content in crawled_pages)
    crawled_urls = [u for u, _ in crawled_pages]

    return combined_markdown, crawled_urls
