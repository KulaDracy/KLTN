from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

# Load environment variables từ file .env
load_dotenv()

def setup_cors(app):
    environment = os.getenv("ENVIRONMENT", "development")
    
    # Có thể lấy origins từ env
    custom_origins = os.getenv("ALLOWED_ORIGINS")
    
    if custom_origins:
        # Nếu có custom origins trong .env
        allowed_origins = custom_origins.split(",")
    elif environment == "production":
        allowed_origins = [
            "https://yourdomain.com",
            "https://www.yourdomain.com"
        ]
    else:
        allowed_origins = [
            "http://localhost:3000",
            "http://localhost:3001",
            "http://127.0.0.1:3000"
        ]
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["Content-Disposition"]
    )