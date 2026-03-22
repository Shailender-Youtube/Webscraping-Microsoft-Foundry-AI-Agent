import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    azure_api_key: str
    azure_endpoint: str
    azure_deployment: str
    azure_api_version: str


def load_config() -> Config:
    api_key = os.getenv("AZURE_FOUNDRY_API_KEY")
    endpoint = os.getenv("AZURE_FOUNDRY_ENDPOINT")
    deployment = os.getenv("AZURE_FOUNDRY_DEPLOYMENT")
    api_version = os.getenv("AZURE_FOUNDRY_API_VERSION")

    missing = [
        name
        for name, val in [
            ("AZURE_FOUNDRY_API_KEY", api_key),
            ("AZURE_FOUNDRY_ENDPOINT", endpoint),
            ("AZURE_FOUNDRY_DEPLOYMENT", deployment),
            ("AZURE_FOUNDRY_API_VERSION", api_version),
        ]
        if not val
    ]

    if missing:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}. "
            "Copy .env.example to .env and fill in the values."
        )

    return Config(
        azure_api_key=api_key,
        azure_endpoint=endpoint,
        azure_deployment=deployment,
        azure_api_version=api_version,
    )
