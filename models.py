from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class FileAnalysisResponse(BaseModel):
    """Response model for file analysis operations."""
    filename: str
    content_type: str
    file_size: int
    file_size_formatted: str
    file_extension: str
    md5_hash: str
    sha256_hash: str
    analyzed_at: str
    analysis_type: str
    is_text: bool
    is_image: bool
    is_document: bool
    is_supported: bool
    metadata: Dict[str, Any]
    text_analysis: Optional[Dict[str, Any]] = None
    image_analysis: Optional[Dict[str, Any]] = None
    document_analysis: Optional[Dict[str, Any]] = None
    binary_analysis: Optional[Dict[str, Any]] = None
    json_analysis: Optional[Dict[str, Any]] = None
    csv_analysis: Optional[Dict[str, Any]] = None
    xml_analysis: Optional[Dict[str, Any]] = None
    content_analysis: Optional[Dict[str, Any]] = None


class ContentAnalysisResult(BaseModel):
    """Model for content analysis results."""
    document_type: Dict[str, Any]
    key_entities: List[Dict[str, str]]
    key_dates: List[str]
    key_numbers: List[Dict[str, Any]]
    summary: str
    sentiment: Dict[str, Any]
    key_phrases: List[str]
    document_metadata: Dict[str, Any]
    specific_details: Optional[Dict[str, Any]] = None


class DocumentTypeResult(BaseModel):
    """Model for document type identification."""
    type: str
    confidence: float
    scores: Dict[str, int]


class EntityResult(BaseModel):
    """Model for named entity extraction."""
    text: str
    label: str
    description: str


class NumberResult(BaseModel):
    """Model for number extraction."""
    value: str
    type: str
    context: str


class SentimentResult(BaseModel):
    """Model for sentiment analysis."""
    label: str
    score: float


class TextAnalysisResult(BaseModel):
    """Model for text analysis results."""
    character_count: int
    word_count: int
    line_count: int
    paragraph_count: int
    average_word_length: float
    unique_words: int
    encoding: str


class ImageAnalysisResult(BaseModel):
    """Model for image analysis results."""
    format: Optional[str] = None
    mode: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    aspect_ratio: Optional[float] = None
    color_depth: Optional[str] = None
    file_size: int
    compression: Optional[str] = None
    error: Optional[str] = None


class BinaryAnalysisResult(BaseModel):
    """Model for binary analysis results."""
    file_size: int
    entropy: float
    null_bytes: int
    printable_ratio: float


class DocumentAnalysisResult(BaseModel):
    """Model for document analysis results."""
    content_type: str
    file_size: int
    note: str


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: Optional[str] = None
    status_code: int


class HealthCheckResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: datetime
    version: str = "1.0.0"
    service: str = "Document Analyzer API"


class SupportedFormatsResponse(BaseModel):
    """Response model for supported file formats."""
    text_formats: List[str]
    image_formats: List[str]
    document_formats: List[str]
    max_file_size_mb: int


class ContentAnalysisSummary(BaseModel):
    """Summary model for content analysis."""
    document_type: str
    confidence: float
    summary: str
    key_details: List[str]
    sentiment: str 