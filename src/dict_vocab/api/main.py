# -*- coding: utf-8 -*-
"""
FastAPI application for dictionary lookup.
"""

import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from dict_vocab.indexer.mdict_indexer import IndexBuilder

app = FastAPI(
    title="Dict Vocab API",
    description="Dictionary lookup API with SQLite optimization",
    version="0.1.0",
)

DEFAULT_DICT_PATH = os.environ.get(
    "DEFAULT_DICT_PATH",
    str(
        Path(__file__).parent.parent.parent.parent
        / "resource"
        / "cobuild2024"
        / "cobuild2024.mdx"
    ),
)

_dict_cache: dict[str, IndexBuilder] = {}


class LookupRequest(BaseModel):
    word: str
    dict_path: Optional[str] = None
    ignorecase: bool = False


class LookupResponse(BaseModel):
    word: str
    definitions: list[str]
    dict_title: Optional[str] = None


class DictInfo(BaseModel):
    path: str
    title: str
    encoding: str


def get_dict_builder(dict_path: str, force_rebuild: bool = False) -> IndexBuilder:
    """Get or create IndexBuilder for given dict path."""
    cache_key = f"{dict_path}:{force_rebuild}"

    if cache_key not in _dict_cache:
        if not os.path.exists(dict_path):
            raise HTTPException(
                status_code=404, detail=f"Dictionary not found: {dict_path}"
            )
        _dict_cache[cache_key] = IndexBuilder(
            fname=dict_path,
            force_rebuild=force_rebuild,
            sql_index=True,
            check=False,
        )

    return _dict_cache[cache_key]


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/dicts", response_model=list[DictInfo])
async def list_dicts():
    """
    List available dictionaries.
    If DEFAULT_DICT_PATH is set, returns info about that dict.
    """
    dicts = []

    # Read at runtime to support environment variable changes
    current_dict_path = os.environ.get("DEFAULT_DICT_PATH", "") or DEFAULT_DICT_PATH

    if current_dict_path and os.path.exists(current_dict_path):
        builder = get_dict_builder(current_dict_path)
        dicts.append(
            DictInfo(
                path=current_dict_path, title=builder.title, encoding=builder.encoding
            )
        )

    return dicts


@app.post("/lookup", response_model=LookupResponse)
async def lookup_word(request: LookupRequest):
    """
    Look up a word in the dictionary.

    Request body:
        word: The word to look up
        dict_path: Optional path to dictionary (uses DEFAULT_DICT_PATH if not provided)
        ignorecase: Whether to ignore case (default: False)
    """
    # Read at runtime to support environment variable changes
    current_default = os.environ.get("DEFAULT_DICT_PATH", "") or DEFAULT_DICT_PATH
    dict_path = request.dict_path or current_default

    if not dict_path:
        raise HTTPException(
            status_code=400,
            detail="No dictionary path provided. Set DEFAULT_DICT_PATH or provide dict_path in request.",
        )

    if not os.path.exists(dict_path):
        raise HTTPException(
            status_code=404, detail=f"Dictionary not found: {dict_path}"
        )

    try:
        builder = get_dict_builder(dict_path)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to load dictionary: {str(e)}"
        )

    try:
        definitions = builder.mdx_lookup(request.word, ignorecase=request.ignorecase)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lookup failed: {str(e)}")

    return LookupResponse(
        word=request.word, definitions=definitions, dict_title=builder.title
    )


@app.get("/resource/{path:path}")
async def serve_resource(path: str):
    """Serve dictionary resource files (CSS, images, audio, etc.)."""
    import os.path

    dict_path = os.environ.get("DEFAULT_DICT_PATH", "") or DEFAULT_DICT_PATH

    print(f"DEBUG: Requested resource: {path}")
    print(
        f"DEBUG: dict_path from env: {os.environ.get('DEFAULT_DICT_PATH', 'NOT SET')}"
    )
    print(f"DEBUG: DEFAULT_DICT_PATH module: {DEFAULT_DICT_PATH}")

    resource_base_dir = None
    resource_subpath = path

    if dict_path:
        dict_dir = Path(dict_path).parent.resolve()
        print(f"DEBUG: dict_dir: {dict_dir}")
        print(f"DEBUG: dict_dir exists: {dict_dir.exists()}")
        if dict_dir.exists():
            resource_base_dir = dict_dir
            # path is like "cobuild2024/cobuild2024.css", extract the subpath
            # dict_dir is already "resource/cobuild2024", so use path as-is
            # but we need to handle if path starts with dict name
            dict_name = dict_dir.name
            if path.startswith(dict_name + "/"):
                resource_subpath = path[len(dict_name) + 1 :]

    # Fallback: auto-detect resource directory from project root
    if resource_base_dir is None:
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent.parent
        fallback_resource_dir = project_root / "resource"
        print(f"DEBUG: fallback_resource_dir: {fallback_resource_dir}")
        print(f"DEBUG: fallback exists: {fallback_resource_dir.exists()}")
        if fallback_resource_dir.exists():
            resource_base_dir = fallback_resource_dir

    print(f"DEBUG: resource_base_dir: {resource_base_dir}")
    print(f"DEBUG: resource_subpath: {resource_subpath}")

    if resource_base_dir is None:
        raise HTTPException(status_code=404, detail="No dictionary configured")

    # Security check: prevent path traversal attacks
    # Resolve the resource path and ensure it's within base directory
    resource_path = (resource_base_dir / resource_subpath).resolve()

    # Security check: ensure resolved path is within base directory
    try:
        base_str = str(resource_base_dir.resolve())
        resource_str = str(resource_path)
        # Ensure both paths use same separators and case for comparison
        base_str = os.path.normcase(os.path.normpath(base_str))
        resource_str = os.path.normcase(os.path.normpath(resource_str))
        # Ensure base path ends with separator for accurate prefix matching
        if not base_str.endswith(os.sep):
            base_str += os.sep
        # Check if resource path starts with base path
        if not resource_str.startswith(base_str):
            raise ValueError("Path traversal detected")
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")

    if not resource_path.exists() or not resource_path.is_file():
        raise HTTPException(status_code=404, detail=f"Resource not found: {path}")

    # Determine content type
    content_type = None
    suffix = resource_path.suffix.lower()
    if suffix == ".css":
        content_type = "text/css"
    elif suffix in [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"]:
        content_type = f"image/{suffix[1:]}"
    elif suffix in [".mp3", ".wav", ".ogg", ".m4a"]:
        content_type = "audio/mpeg"
    elif suffix == ".js":
        content_type = "application/javascript"
    elif suffix == ".html":
        content_type = "text/html"

    return FileResponse(str(resource_path), media_type=content_type)


# Get the directory containing this file
CURRENT_DIR = Path(__file__).parent
STATIC_DIR = CURRENT_DIR / "static"

# Mount static files after all API routes are defined
# This ensures API routes take precedence over static files
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
