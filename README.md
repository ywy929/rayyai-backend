# RayyAI Backend

Personal financial tracker and analyzer API with AI-powered insights.

## Tech Stack

- **Framework**: FastAPI
- **Database**: PostgreSQL (primary), MongoDB Atlas (chat messages)
- **ORM**: SQLAlchemy
- **AI**: Google Gemini 2.0 Flash (chat, document processing, insights)
- **Storage**: AWS S3 (production) / Local filesystem (development)
- **Authentication**: JWT (JSON Web Tokens)

## Project Structure

```
rayyai-backend/
├── main.py                 # FastAPI app entry point
├── database.py             # PostgreSQL + MongoDB connection
├── models.py               # SQLAlchemy ORM models
├── schemas.py              # Pydantic request/response schemas
├── routers/                # API endpoints
│   ├── auth.py             # Authentication (login, register, /me)
│   ├── users.py            # User CRUD
│   ├── accounts.py         # Account management
│   ├── transactions.py     # Income/Expense CRUD
│   ├── budgets.py          # Budget management
│   ├── goals.py            # Financial goals
│   ├── cards.py            # Credit card recommendations
│   ├── statements.py       # Statement upload & management
│   ├── scanner.py          # Receipt image scanning
│   ├── chat.py             # AI chat with RAG
│   ├── insights.py         # AI financial insights
│   └── utils.py            # Auth, S3, helpers
├── services/               # Business logic
│   ├── processors/         # Document processors
│   │   ├── statement_processor.py  # Bank/CC statement extraction
│   │   └── ctos_processor.py       # CTOS credit report extraction
│   ├── gemini_service.py   # Gemini API wrapper
│   ├── rag_service.py      # RAG context retrieval
│   ├── embedding_service.py # Text embeddings
│   ├── pii_masking.py      # PII detection/masking
│   ├── context_summarizer.py # Conversation summarization
│   ├── conversation_manager.py # Chat history management
│   ├── action_executor.py  # Execute AI-suggested actions
│   ├── search_service.py   # Full-text search
│   └── search_setup.py     # PostgreSQL FTS setup
└── uploads/                # Local file storage (dev)
    ├── statements/         # Statement PDFs
    └── ctos/               # CTOS report PDFs
```

## Database Models

| Model | Description |
|-------|-------------|
| `User` | User accounts with soft delete |
| `Account` | Bank, credit card, e-wallet, cash accounts |
| `AccountBalanceSnapshot` | Balance snapshots for performance |
| `Income` | Income transactions |
| `Expense` | Expense transactions |
| `Transfer` | Transfers between accounts |
| `Statement` | Uploaded statement metadata |
| `Budget` | Budget tracking by category |
| `Goal` | Financial goals with progress |
| `UserCreditCard` | User's credit cards |
| `MarketCreditCard` | Credit card marketplace |
| `ChatConversation` | AI chat conversations |
| `ChatMessage` | Individual chat messages |
| `ContextSummary` | Conversation summaries |
| `AIAnalysis` | AI analysis results |

## API Endpoints

### Authentication (`/auth`)
- `POST /auth/register` - Register new user
- `POST /auth/login` - Login, returns JWT
- `GET /auth/me` - Get current user info
- `POST /auth/logout` - Logout

### Users (`/users`)
- `GET /users` - Get user profile
- `PUT /users` - Update user profile

### Accounts (`/accounts`)
- `GET /accounts` - List user accounts
- `POST /accounts` - Create account
- `GET /accounts/{id}` - Get account details
- `PUT /accounts/{id}` - Update account
- `DELETE /accounts/{id}` - Delete account
- `GET /accounts/total-balance` - Get total balance

### Transactions (`/transactions`)
- `GET /transactions` - List transactions (with filters)
- `POST /transactions/income` - Create income
- `POST /transactions/expense` - Create expense
- `PUT /transactions/{type}/{id}` - Update transaction
- `DELETE /transactions/{type}/{id}` - Delete transaction

### Budgets (`/budgets`)
- `GET /budgets` - List budgets
- `POST /budgets` - Create budget
- `PUT /budgets/{id}` - Update budget
- `DELETE /budgets/{id}` - Delete budget

### Goals (`/goals`)
- `GET /goals` - List financial goals
- `POST /goals` - Create goal
- `PUT /goals/{id}` - Update goal
- `DELETE /goals/{id}` - Delete goal

### Statements
- `POST /statements/upload` - Upload bank statement PDF
- `POST /statements/upload-ctos` - Upload CTOS report
- `GET /statements` - List statements
- `GET /statements/{id}` - Get statement details
- `DELETE /statements/{id}` - Delete statement

### Scanner (`/scanner`)
- `POST /scanner/receipt` - Scan receipt image

### Chat (`/chat`)
- `POST /chat/message` - Send message to AI
- `GET /chat/conversations` - List conversations
- `GET /chat/conversations/{id}` - Get conversation
- `DELETE /chat/conversations/{id}` - Delete conversation

### Insights
- `GET /insights/financial-health` - Get financial health score
- `GET /insights/spending-analysis` - Get spending analysis

### Cards (`/cards`)
- `GET /cards/recommendations` - Get credit card recommendations
- `GET /cards/market` - Browse credit card marketplace

## AI Features

### Document Processing
- **Statement Processor**: Extracts transactions from bank/credit card/e-wallet PDFs using Gemini Vision
- **CTOS Processor**: Extracts credit score and details from CTOS reports
- **Receipt Scanner**: Extracts merchant, amount, date from receipt images

### Conversational AI
- **RAG Service**: Retrieves user's financial context (accounts, transactions, budgets, goals)
- **Gemini Service**: Handles chat with 1M token context window
- **Action Executor**: Executes AI-suggested actions (create budget, add transaction)
- **PII Masking**: Masks sensitive data before sending to AI
- **Context Summarizer**: Summarizes long conversations

## Getting Started

### Prerequisites
- Python 3.11+
- PostgreSQL 14+
- MongoDB Atlas account (optional, for chat)
- Google Gemini API key
- AWS S3 bucket (optional, for production)

### Installation

```bash
# Clone repository
cd rayyai-backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Create `.env` file:

```env
# Database
DB_USER=postgres
DB_PASSWORD=your_password
DB_HOST=localhost
DB_PORT=5432
DB_NAME=rayyai

# JWT
SECRET_KEY=your-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=1440

# Google Gemini AI
GEMINI_API_KEY=your-gemini-api-key

# MongoDB (optional)
MONGODB_ATLAS_CLUSTER_URI=mongodb+srv://...
MONGODB_DB_NAME=rayyai

# AWS S3 (optional, for production)
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
S3_BUCKET_NAME=your-bucket
AWS_REGION=ap-southeast-1

# File storage
UPLOAD_DIR=uploads
BASE_URL=http://localhost:8000
```

### Running the Server

```bash
# Development
uvicorn main:app --reload --port 8000

# Production
uvicorn main:app --host 0.0.0.0 --port 8000
```

### API Documentation

Once running, access:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Deployment

### Docker

```bash
docker build -t rayyai-backend .
docker run -p 8000:8000 --env-file .env rayyai-backend
```

### Google Cloud Run

The app is configured for Google Cloud Run with Cloud SQL connector:
- Set `ENVIRONMENT=cloud` in environment
- Configure `INSTANCE_CONNECTION_NAME` for Cloud SQL

## CORS Configuration

Allowed origins:
- `http://localhost:5173` (development)
- `https://fir-tutorial-d397a.web.app` (production)
