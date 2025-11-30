from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import models
from database import engine
from routers import auth, users, accounts, transactions, budgets, goals, cards, statements, scanner, chat, insights
from services.search_setup import ensure_chat_message_fts

# Create tables
models.Base.metadata.create_all(bind=engine)
# Ensure FTS is configured for chat messages
try:
    ensure_chat_message_fts(engine)
except Exception:
    # Non-fatal if extension/privileges are missing; API still works without search
    pass

app = FastAPI(
    title="RayyAI API",
    version="1.0.0",
    description="Personal financial tracker and analyser",
)

# CORS middleware - configured for local development
# Note: Cannot use "*" with allow_credentials=True, so we list origins explicitly
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://fir-tutorial-d397a.web.app"  # <--- Slash removed here
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

# Include routers
app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
app.include_router(users.router, prefix="/users", tags=["Users"])
app.include_router(accounts.router, prefix="/accounts", tags=["Accounts"])
app.include_router(transactions.router, prefix="/transactions", tags=["Transactions"])
app.include_router(budgets.router, prefix="/budgets", tags=["Budgets"])
app.include_router(goals.router, prefix="/goals", tags=["Goals"])
app.include_router(cards.router, prefix="/cards", tags=["Credit Cards"])
app.include_router(statements.router, tags=["Statements"])
app.include_router(scanner.router, prefix="/scanner", tags=["Scanner"])
app.include_router(chat.router, prefix="/chat", tags=["Chat"])
app.include_router(insights.router, tags=["Insights"])

# Mount static file serving for uploaded statements
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
if os.path.exists(UPLOAD_DIR):
    app.mount("/files", StaticFiles(directory=UPLOAD_DIR), name="files")

@app.get("/")
async def root():
    return {"message": "RayyAI API", "version": "1.0.0", "docs": "/docs"}