from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application Configuration
    app_name: str = "Document Analyzer API"
    debug: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    
    # File analysis configuration
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_extensions: list = [
        ".pdf", ".doc", ".docx", ".txt", ".jpg", ".jpeg", ".png", 
        ".gif", ".bmp", ".tiff", ".webp", ".csv", ".json", ".xml",
        ".html", ".css", ".js", ".zip", ".rar"
    ]
    
    # Analysis settings
    enable_image_analysis: bool = True
    enable_text_analysis: bool = True
    enable_document_analysis: bool = True
    enable_binary_analysis: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "allow"


# Create settings instance
settings = Settings() 