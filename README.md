# Sales Intelligence Pipeline

An AI-powered web application that takes a company name and delivers a full structured sales brief in under 60 seconds. Built with **Crawl4AI**, **Azure AI Foundry**, and **FastAPI**.

---

## How It Works

1. You type a company name into the web portal
2. An Azure AI Agent searches the web and returns matching companies
3. You select the company you want
4. Crawl4AI crawls up to 17 pages of their website
5. Azure AI analyses the content and extracts a structured sales brief
6. The brief is displayed in the portal and can be downloaded as a Markdown file

---

## Prerequisites

- Python 3.10 or higher
- [Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli) installed
- An Azure AI Foundry project with:
  - A deployed GPT model
  - An Agent named `crawler-agent` (version `1`) with the `web_search` tool enabled

---

## Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd webscraper
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Crawl4AI browser (one-time)

Crawl4AI uses a headless Chromium browser. Install it with:

```bash
crawl4ai-setup
```

### 5. Configure environment variables

Create a `.env` file in the project root:

```env
AZURE_FOUNDRY_ENDPOINT=https://<your-hub>.services.ai.azure.com/api/projects/<your-project>/openai/v1
AZURE_FOUNDRY_DEPLOYMENT=gpt-5.4-mini
AZURE_FOUNDRY_API_VERSION=2024-02-01
```

> **Note:** This project uses Azure AD authentication (not API keys). No API key is needed in the `.env`.

### 6. Log in to Azure

```bash
az login --tenant <your-tenant-id>
```

Make sure you are logged into the correct tenant that owns your Azure AI Foundry resource.

---

## Running the App

### Web Portal (recommended)

```bash
python app.py
```

Then open your browser at:

```
http://localhost:8000
```

### Command Line (optional)

Search interactively and generate a brief from the terminal:

```bash
python main.py
```

To pass a URL directly and save the output:

```bash
python main.py --url https://www.example.com --output brief.md
```

---

## Project Structure

```
webscraper/
├── app.py              # FastAPI web server
├── main.py             # CLI entry point
├── agent_search.py     # Azure AI Agent — company search via web_search tool
├── crawler.py          # Crawl4AI — crawls up to 17 pages of a website
├── extractor.py        # Azure AI — analyses content and returns SalesBrief
├── models.py           # Pydantic model — SalesBrief with 13 fields
├── config.py           # Loads .env configuration
├── templates/
│   └── index.html      # Single-page web frontend
├── requirements.txt
└── .env                # Your credentials (not committed to git)
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `crawl4ai` | Async web crawler with Playwright/Chromium |
| `openai` | Azure AI Foundry API client |
| `azure-identity` | Azure AD authentication (`DefaultAzureCredential`) |
| `azure-ai-projects` | Azure AI Foundry Agent client |
| `fastapi` | Web framework |
| `uvicorn` | ASGI server |
| `pydantic` | Data model validation |
| `python-dotenv` | Load `.env` file |
