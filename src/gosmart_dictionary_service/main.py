"""FastAPI application entry point for the dictionary service."""

from fastapi import FastAPI

from gosmart_dictionary_service import __version__

app = FastAPI(
    title="Gosmart.Health Dictionary Service",
    version=__version__,
)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
