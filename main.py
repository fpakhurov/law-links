"""Law Links Service: extracts legal references from Russian text.

Run locally:
    python main.py
    uvicorn main:app --host 0.0.0.0 --port 8978 --reload

Swagger UI: http://localhost:8978/docs
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import Depends, FastAPI, Request

from law_links import DEFAULT_ALIASES_PATH, DEFAULT_CHAIN_MODEL_PATH
from law_links.extractor import Extractor, LinkExtractor
from law_links.schemas import LinksResponse, TextRequest

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("law_links.service")

ALIASES_PATH = Path(os.getenv("LAW_ALIASES_PATH", str(DEFAULT_ALIASES_PATH)))
CHAIN_MODEL_PATH = Path(os.getenv("CHAIN_MODEL_PATH", str(DEFAULT_CHAIN_MODEL_PATH)))


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting service, aliases: %s, chain model: %s", ALIASES_PATH, CHAIN_MODEL_PATH)
    app.state.extractor = Extractor.from_files(ALIASES_PATH, CHAIN_MODEL_PATH)
    yield
    logger.info("Service stopped")


def get_extractor(request: Request) -> LinkExtractor:
    return request.app.state.extractor


app = FastAPI(
    title="Law Links Service",
    description="Сервис для выделения юридических ссылок из текста",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/detect")
async def get_law_links(
    data: TextRequest,
    extractor: LinkExtractor = Depends(get_extractor),
) -> LinksResponse:
    """Принимает текст и возвращает список юридических ссылок."""
    return LinksResponse(links=extractor.extract(data.text))


@app.get("/health")
async def health_check():
    """Проверка состояния сервиса."""
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8978)
