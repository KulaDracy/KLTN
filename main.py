from fastapi import FastAPI
import uvicorn

from core.config import setup_cors

from api.scan_router import router as scan_router
from api.mark_router import router as mark_router


def create_app():
    app = FastAPI(title="PDF Scan + OCR API")

    # CORS
    setup_cors(app)
    
    # Routers
    app.include_router(mark_router, prefix="/api")
    app.include_router(scan_router, prefix="/api/scan", tags=["Scan PDF"])
    app.include_router(mark_router, prefix="/api/mark", tags=["Mark Pages"])

    @app.get("/")
    def home():
        return {"message": "Welcome to FastAPI PDF API!"}

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
