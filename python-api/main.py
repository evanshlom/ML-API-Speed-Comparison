from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router
import uvicorn

app = FastAPI(title="Python ONNX API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)