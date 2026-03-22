import json
import logging
from openai import OpenAI
from azure.identity import DefaultAzureCredential

from config import Config
from models import SalesBrief

logger = logging.getLogger(__name__)

MAX_WORDS = 12_000

SYSTEM_PROMPT = (
    "You are a B2B sales intelligence analyst. Your job is to read raw website content "
    "and extract a structured brief that a presales consultant will use to prepare for a "
    "first meeting with this company. Be specific and factual. Only include what is supported "
    "by the content. Do not invent information."
)


def _build_user_prompt(markdown: str) -> str:
    return (
        f"Read the following website content and extract a sales intelligence brief.\n\n"
        f"Return ONLY a valid JSON object matching this schema — no explanation, "
        f"no markdown code fences, no extra text:\n\n"
        f"{json.dumps(SalesBrief.model_json_schema(), indent=2)}\n\n"
        f"Website content:\n\n{markdown}"
    )


def _truncate(markdown: str) -> str:
    words = markdown.split()
    if len(words) > MAX_WORDS:
        logger.warning(
            f"Content exceeds {MAX_WORDS} words ({len(words)} words). Truncating."
        )
        return " ".join(words[:MAX_WORDS])
    return markdown


def _parse_response(raw: str) -> SalesBrief:
    try:
        parsed = json.loads(raw)
        return SalesBrief(**parsed)
    except (json.JSONDecodeError, Exception):
        return None


def extract_sales_brief(markdown: str, config: Config) -> SalesBrief:
    # Azure AI Foundry endpoints may disable key auth and require Azure AD tokens
    try:
        credential = DefaultAzureCredential()
        token = credential.get_token("https://ai.azure.com/.default")
        api_key = token.token
        logger.info("Using Azure AD token authentication.")
    except Exception:
        logger.info("Azure AD auth unavailable, falling back to API key.")
        api_key = config.azure_api_key

    client = OpenAI(
        api_key=api_key,
        base_url=config.azure_endpoint,
    )

    markdown = _truncate(markdown)
    user_prompt = _build_user_prompt(markdown)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    response = client.chat.completions.create(
        model=config.azure_deployment,
        messages=messages,
    )
    raw = response.choices[0].message.content.strip()

    brief = _parse_response(raw)
    if brief:
        return brief

    # Retry once with explicit reminder
    logger.warning("JSON parsing failed on first attempt. Retrying with explicit reminder.")
    messages.append({"role": "assistant", "content": raw})
    messages.append({
        "role": "user",
        "content": (
            "Your response was not valid JSON. Please return ONLY the JSON object — "
            "no explanation, no markdown code fences, nothing else."
        ),
    })

    retry_response = client.chat.completions.create(
        model=config.azure_deployment,
        messages=messages,
    )
    raw_retry = retry_response.choices[0].message.content.strip()

    brief = _parse_response(raw_retry)
    if brief:
        return brief

    raise ValueError(
        f"Failed to parse JSON response after retry.\nRaw response:\n{raw_retry}"
    )
