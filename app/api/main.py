"""FastAPI backend for the Training Data Curation Bot."""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from training_data_bot import TrainingDataBot
from training_data_bot.core.config import settings
from training_data_bot.tasks.task_manager import TaskManager

# Initialize FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    description="FastAPI backend for the Training Data Curation Bot - Generate training data from documents",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global bot instance (in production, use proper session management)
bot: TrainingDataBot | None = None


def get_bot() -> TrainingDataBot:
    """Get or create the bot instance."""
    global bot
    if bot is None:
        bot = TrainingDataBot()
    return bot


# Pydantic Models for Request/Response


class LoadDocumentsRequest(BaseModel):
    """Request model for loading documents."""

    sources: list[str] = Field(..., description="List of file paths or URLs to load")
    max_concurrent: int = Field(default=4, description="Max concurrent loads")


class ProcessDocumentsRequest(BaseModel):
    """Request model for processing documents."""

    sources: list[str] | None = Field(default=None, description="Sources to process")
    task_types: list[str] = Field(..., description="Task types to generate")
    quality_filter: bool = Field(default=True, description="Enable quality filtering")
    chunk_size: int = Field(default=1200, description="Text chunk size")
    chunk_overlap: int = Field(default=150, description="Text chunk overlap")


class ExportDatasetRequest(BaseModel):
    """Request model for exporting dataset."""

    dataset_id: str = Field(..., description="Dataset ID to export")
    output_path: str = Field(..., description="Output file path")
    format: str = Field(default="jsonl", description="Export format (json, jsonl, csv, zip)")


class EvaluateDatasetRequest(BaseModel):
    """Request model for evaluating dataset."""

    dataset_id: str = Field(..., description="Dataset ID to evaluate")


class DocumentInfo(BaseModel):
    """Document information model."""

    id: str
    source: str
    doc_type: str
    status: str
    char_count: int = 0


class DatasetInfo(BaseModel):
    """Dataset information model."""

    id: str
    example_count: int
    created_at: str
    task_types: list[str]


class JobInfo(BaseModel):
    """Job information model."""

    id: str
    status: str
    created_at: str
    completed_at: str | None = None


# API Endpoints


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Training Data Curation Bot API",
        "version": "1.0.0",
        "environment": settings.APP_ENV,
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "app_name": settings.APP_NAME,
        "environment": settings.APP_ENV,
    }


@app.get("/bot-info")
async def bot_info():
    """Get bot information."""
    return {
        "bot_class": "TrainingDataBot",
        "message": "Training Data Bot is ready",
    }


@app.get("/tasks/supported")
async def get_supported_tasks():
    """Get list of supported task types."""
    task_manager = TaskManager()
    return {
        "tasks": task_manager.get_supported_tasks(),
        "configs": task_manager.TASK_CONFIGS,
    }


@app.post("/documents/load")
async def load_documents(request: LoadDocumentsRequest):
    """Load documents from sources."""
    try:
        bot_instance = get_bot()
        documents = await bot_instance.load_documents(request.sources)

        return {
            "status": "success",
            "documents_loaded": len(documents),
            "documents": [
                {
                    "id": doc.get("id"),
                    "source": doc.get("source"),
                    "doc_type": doc.get("doc_type"),
                    "status": doc.get("status"),
                    "char_count": doc.get("metadata", {}).get("char_count", 0),
                }
                for doc in documents
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a document file."""
    try:
        # Create temp directory
        temp_dir = tempfile.mkdtemp()

        # Save uploaded file
        file_path = os.path.join(temp_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Load document
        bot_instance = get_bot()
        documents = await bot_instance.load_documents([file_path])

        # Clean up temp file
        shutil.rmtree(temp_dir)

        if not documents:
            raise HTTPException(status_code=400, detail="Failed to load document")

        doc = documents[0]

        return {
            "status": "success",
            "document": {
                "id": doc.get("id"),
                "source": doc.get("source"),
                "doc_type": doc.get("doc_type"),
                "status": doc.get("status"),
                "char_count": doc.get("metadata", {}).get("char_count", 0),
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents")
async def list_documents():
    """List all loaded documents."""
    try:
        bot_instance = get_bot()
        documents = bot_instance.list_documents()
        return {"documents": documents, "count": len(documents)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents/{document_id}")
async def get_document(document_id: str):
    """Get a specific document."""
    try:
        bot_instance = get_bot()
        document = bot_instance.get_document(document_id)

        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        return {"document": document}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process")
async def process_documents(request: ProcessDocumentsRequest):
    """Process documents and generate training data."""
    try:
        bot_instance = get_bot()

        # Load documents if sources provided
        if request.sources:
            documents = await bot_instance.load_documents(request.sources)
        else:
            documents = None

        # Process documents
        result = await bot_instance.process_documents(
            documents=documents,
            task_types=request.task_types,
            quality_filter=request.quality_filter,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
        )

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/datasets")
async def list_datasets():
    """List all datasets."""
    try:
        bot_instance = get_bot()
        datasets = bot_instance.list_datasets()
        return {"datasets": datasets, "count": len(datasets)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/datasets/{dataset_id}")
async def get_dataset(dataset_id: str):
    """Get a specific dataset."""
    try:
        bot_instance = get_bot()
        dataset = bot_instance.get_dataset(dataset_id)

        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")

        return {
            "dataset": {
                "id": dataset.get("id"),
                "created_at": dataset.get("created_at"),
                "task_types": dataset.get("task_types"),
                "document_count": dataset.get("document_count"),
                "example_count": len(dataset.get("examples", [])),
                "examples": dataset.get("examples", [])[:10],  # Return first 10
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/datasets/export")
async def export_dataset(request: ExportDatasetRequest):
    """Export a dataset to file."""
    try:
        bot_instance = get_bot()

        result = await bot_instance.export_dataset(
            dataset=request.dataset_id,
            output_path=request.output_path,
            format=request.format,
        )

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/datasets/evaluate")
async def evaluate_dataset(request: EvaluateDatasetRequest):
    """Evaluate a dataset."""
    try:
        bot_instance = get_bot()
        dataset = bot_instance.get_dataset(request.dataset_id)

        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")

        result = await bot_instance.evaluate_dataset(dataset)

        return {
            "dataset_id": request.dataset_id,
            "evaluation": result,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/jobs")
async def list_jobs():
    """List all jobs."""
    try:
        bot_instance = get_bot()
        jobs = bot_instance.list_jobs()
        return {"jobs": jobs, "count": len(jobs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """Get a specific job."""
    try:
        bot_instance = get_bot()
        job = bot_instance.get_job(job_id)

        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        return {"job": job}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global bot
    if bot:
        await bot.cleanup()
        bot = None
