# -*- coding: utf-8 -*-
"""
FastAPI application for dictionary lookup.
"""

import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from dict_vocab.indexer.mdict_indexer import IndexBuilder

app = FastAPI(
    title="Dict Vocab API",
    description="Dictionary lookup API with SQLite optimization",
    version="0.1.0"
)

DEFAULT_DICT_PATH = os.environ.get("DEFAULT_DICT_PATH", "")

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
            raise HTTPException(status_code=404, detail=f"Dictionary not found: {dict_path}")
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
    
    if DEFAULT_DICT_PATH and os.path.exists(DEFAULT_DICT_PATH):
        builder = get_dict_builder(DEFAULT_DICT_PATH)
        dicts.append(DictInfo(
            path=DEFAULT_DICT_PATH,
            title=builder.title,
            encoding=builder.encoding
        ))
    
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
    dict_path = request.dict_path or DEFAULT_DICT_PATH
    
    if not dict_path:
        raise HTTPException(
            status_code=400,
            detail="No dictionary path provided. Set DEFAULT_DICT_PATH or provide dict_path in request."
        )
    
    if not os.path.exists(dict_path):
        raise HTTPException(status_code=404, detail=f"Dictionary not found: {dict_path}")
    
    try:
        builder = get_dict_builder(dict_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load dictionary: {str(e)}")
    
    try:
        definitions = builder.mdx_lookup(request.word, ignorecase=request.ignorecase)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lookup failed: {str(e)}")
    
    return LookupResponse(
        word=request.word,
        definitions=definitions,
        dict_title=builder.title
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
