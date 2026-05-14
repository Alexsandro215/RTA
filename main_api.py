from fastapi import FastAPI

from app.web.routes import router


app = FastAPI(title="RTA")
app.include_router(router)
