import json
import logging
import re
from typing import List, Dict

from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient

logger = logging.getLogger(__name__)

AGENT_ENDPOINT = "https://XXXX.services.ai.azure.com/api/projects/XXXX"
AGENT_NAME = "crawler-agent"
AGENT_VERSION = "1"


def _extract_json(text: str) -> str:
    """Strip markdown fences and extract the JSON array from the response."""
    # Try to find a JSON array directly
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        return match.group(0)
    return text.strip()


def search_company_options(query: str) -> List[Dict]:
    """
    Use the Azure AI Agent (with web_search) to find matching company websites.
    Returns a list of dicts: [{name, url, description}, ...]
    """
    client = AIProjectClient(
        endpoint=AGENT_ENDPOINT,
        credential=DefaultAzureCredential(),
    )
    openai_client = client.get_openai_client()

    prompt = (
        f'Search the web for the company "{query}" and find their official website. '
        f'Return up to 3 matching companies as a JSON array in this exact format:\n'
        f'[{{"name": "Company Full Name", "url": "https://www.company.com", "description": "One line description of what they do"}}]\n'
        f'Return ONLY the JSON array — no explanation, no markdown code fences, nothing else.'
    )

    logger.info(f"Agent searching for: {query}")
    response = openai_client.responses.create(
        input=[{"role": "user", "content": prompt}],
        extra_body={
            "agent_reference": {
                "name": AGENT_NAME,
                "version": AGENT_VERSION,
                "type": "agent_reference",
            }
        },
    )

    raw = response.output_text.strip()
    cleaned = _extract_json(raw)

    try:
        options = json.loads(cleaned)
        return [o for o in options if o.get("url") and o.get("name")]
    except json.JSONDecodeError:
        logger.error(f"Could not parse agent response as JSON:\n{raw}")
        raise ValueError(f"Agent returned unexpected format. Raw response:\n{raw}")
