from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import router  # type: ignore
from backend.core.container import AppContainer


@asynccontextmanager
async def lifespan(app: FastAPI):
    container = getattr(app.state, 'container', None) or AppContainer()
    app.state.container = container
    try:
        container.sections.load()
    except Exception:  # noqa: BLE001, S110
        pass
    container.faiss.load()
    container.neo4j.connect()
    yield
    container.neo4j.close()


def create_app(container: AppContainer | None = None) -> FastAPI:
    app = FastAPI(title='LegalVerse API', version='0.1.0', lifespan=lifespan)
    if container is not None:
        app.state.container = container
    app.add_middleware(CORSMiddleware, allow_origins=[
                       '*'], allow_methods=['*'], allow_headers=['*'])
    app.include_router(router, prefix='/api')
    return app


app = create_app()
