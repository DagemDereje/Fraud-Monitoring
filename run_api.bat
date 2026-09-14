@echo off
call .venv\Scripts\activate
uvicorn src.app:app --reload
