from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.web.routes import router, start_paper_trading_worker


@asynccontextmanager
async def lifespan(app: FastAPI):
    start_paper_trading_worker()
    yield


app = FastAPI(title="RTA", lifespan=lifespan)
app.include_router(router)
