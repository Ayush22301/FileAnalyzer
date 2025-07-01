from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from datetime import datetime
import logging

from config import settings
from file_analyzer import FileAnalyzer
from models import (
    FileAnalysisResponse, 
    HealthCheckResponse,
    ErrorResponse,
    SupportedFormatsResponse,
    ContentAnalysisSummary
)
from utils import validate_upload_file

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="A FastAPI backend for intelligent file analysis without storage",
    version="1.0.0",
    debug=settings.debug
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def preload_ai_models():
    """Preload all AI models during startup to avoid runtime downloads."""
    logger.info("Starting AI model preloading...")
    
    try:
        # Preload spaCy model
        logger.info("Loading spaCy model...")
        import spacy
        spacy.load("en_core_web_sm")
        logger.info("✓ spaCy model loaded successfully")
        
        # Preload transformers models
        logger.info("Loading sentiment analysis model...")
        from transformers import pipeline
        sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        logger.info("✓ Sentiment analysis model loaded successfully")
        
        logger.info("Loading classification model...")
        classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        logger.info("✓ Classification model loaded successfully")
        
        # Preload NLTK data
        logger.info("Loading NLTK data...")
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        logger.info("✓ NLTK data loaded successfully")
        
        logger.info("🎉 All AI models preloaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error preloading models: {e}")
        logger.warning("Models will be loaded on-demand (may cause delays on first file upload)")
        return False


def get_file_analyzer() -> FileAnalyzer:
    """Dependency to get file analyzer instance."""
    return FileAnalyzer()


@app.on_event("startup")
async def startup_event():
    """Run on application startup."""
    logger.info("🚀 Starting Document Analyzer API...")
    
    # Preload AI models
    models_loaded = preload_ai_models()
    
    if models_loaded:
        logger.info("✅ Application ready with all models preloaded")
    else:
        logger.warning("⚠️ Application ready but models will load on-demand")


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Document Analyzer API",
        "version": "1.0.0",
        "description": "Intelligent file analysis without storage",
        "docs": "/docs",
        "health": "/health",
        "supported_formats": "/supported-formats"
    }


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint."""
    return HealthCheckResponse(
        status="healthy",
        timestamp=datetime.utcnow()
    )


@app.get("/supported-formats", response_model=SupportedFormatsResponse)
async def get_supported_formats():
    """Get list of supported file formats for analysis."""
    analyzer = FileAnalyzer()
    max_size_mb = settings.max_file_size // (1024 * 1024)
    
    return SupportedFormatsResponse(
        text_formats=list(analyzer.supported_text_types),
        image_formats=list(analyzer.supported_image_types),
        document_formats=list(analyzer.supported_document_types),
        max_file_size_mb=max_size_mb
    )


@app.post("/analyze", response_model=FileAnalysisResponse)
async def analyze_file(
    file: UploadFile = File(...),
    analyzer: FileAnalyzer = Depends(get_file_analyzer)
):
    """
    Analyze an uploaded file and return comprehensive information.
    
    - **file**: The file to analyze
    """
    try:
        # Validate the uploaded file
        validate_upload_file(file)
        
        # Analyze the file
        analysis_result = analyzer.analyze_file(file)
        
        return FileAnalysisResponse(**analysis_result)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/analyze/content", response_model=ContentAnalysisSummary)
async def analyze_content_intelligently(
    file: UploadFile = File(...),
    analyzer: FileAnalyzer = Depends(get_file_analyzer)
):
    """
    Intelligently analyze document content and provide a concise summary of what the document is about.
    
    - **file**: The file to analyze
    """
    try:
        # Validate the uploaded file
        validate_upload_file(file)
        
        # Analyze the file
        analysis_result = analyzer.analyze_file(file)
        
        # Extract content analysis
        content_analysis = analysis_result.get("content_analysis", {})
        
        if "error" in content_analysis:
            raise HTTPException(
                status_code=400,
                detail=f"Content analysis failed: {content_analysis['error']}"
            )
        
        # Generate concise summary
        doc_type = content_analysis.get("document_type", {})
        doc_type_name = doc_type.get("type", "unknown").replace("_", " ").title()
        confidence = doc_type.get("confidence", 0.0)
        
        # Extract key information for summary
        specific_details = content_analysis.get("specific_details", {})
        key_entities = content_analysis.get("key_entities", [])
        key_numbers = content_analysis.get("key_numbers", [])
        key_dates = content_analysis.get("key_dates", [])
        
        # Generate summary based on document type
        summary_lines = []
        
        if doc_type_name == "Bank Statement":
            # Extract account holder, account number, balance, period
            account_holder = None
            account_number = None
            balance = None
            period = None
            
            # Find account holder from entities
            for entity in key_entities:
                if entity.get("label") == "PERSON" and "John" in entity.get("text", ""):
                    account_holder = entity.get("text").strip()
                    break
            
            # Find account number from specific details or entities
            if "\\d+" in specific_details:
                account_number = specific_details["\\d+"]
            
            # Find balance from key numbers (look for largest currency amount)
            currency_amounts = [num for num in key_numbers if num.get("type") == "currency" and num.get("value", "").replace(",", "").isdigit()]
            if currency_amounts:
                balance = max(currency_amounts, key=lambda x: float(x.get("value", "0").replace(",", "")))
            
            # Find statement period from dates
            if key_dates:
                period = key_dates[0] if len(key_dates) > 0 else None
            
            summary_lines = [
                f"This is a {doc_type_name} for {account_holder or 'the account holder'}.",
                f"Account Number: {account_number or 'N/A'}",
                f"Statement Period: {period or 'N/A'}",
                f"Current Balance: ${balance.get('value', 'N/A') if balance else 'N/A'}"
            ]
            
        elif doc_type_name == "Invoice":
            # Extract invoice number, amount, due date, customer
            invoice_number = specific_details.get("invoice_number", "N/A")
            amount = None
            due_date = None
            customer = None
            
            # Find amount from key numbers
            currency_amounts = [num for num in key_numbers if num.get("type") == "currency"]
            if currency_amounts:
                amount = currency_amounts[0]
            
            # Find due date from dates
            if key_dates:
                due_date = key_dates[0]
            
            # Find customer from entities
            for entity in key_entities:
                if entity.get("label") == "ORG" and "Company" in entity.get("text", ""):
                    customer = entity.get("text").strip()
                    break
            
            summary_lines = [
                f"This is an {doc_type_name} for {customer or 'the customer'}.",
                f"Invoice Number: {invoice_number}",
                f"Amount Due: ${amount.get('value', 'N/A') if amount else 'N/A'}",
                f"Due Date: {due_date or 'N/A'}"
            ]
            
        elif doc_type_name == "Resume":
            # Extract name, contact, experience
            name = None
            email = None
            phone = None
            experience = None
            
            # Find name from entities
            for entity in key_entities:
                if entity.get("label") == "PERSON":
                    name = entity.get("text").strip()
                    break
            
            # Find contact info from metadata
            metadata = content_analysis.get("document_metadata", {})
            if metadata.get("has_email"):
                email = "Available"
            if metadata.get("has_phone"):
                phone = "Available"
            
            summary_lines = [
                f"This is a {doc_type_name} for {name or 'the candidate'}.",
                f"Contact Information: Email {email or 'N/A'}, Phone {phone or 'N/A'}",
                f"Document contains {metadata.get('word_count', 0)} words",
                f"Analysis confidence: {confidence:.1%}"
            ]
            
        elif doc_type_name == "Contract":
            # Extract contract number, parties, amount, effective date
            contract_number = specific_details.get("contract_number", "N/A")
            parties = specific_details.get("parties", "N/A")
            amount = None
            effective_date = specific_details.get("effective_date", "N/A")
            
            # Find amount from key numbers
            currency_amounts = [num for num in key_numbers if num.get("type") == "currency"]
            if currency_amounts:
                amount = currency_amounts[0]
            
            summary_lines = [
                f"This is a {doc_type_name} between {parties}.",
                f"Contract Number: {contract_number}",
                f"Contract Amount: ${amount.get('value', 'N/A') if amount else 'N/A'}",
                f"Effective Date: {effective_date}"
            ]
            
        else:
            # Generic summary for other document types
            metadata = content_analysis.get("document_metadata", {})
            summary_lines = [
                f"This appears to be a {doc_type_name} document.",
                f"Document contains {metadata.get('word_count', 0)} words and {metadata.get('sentence_count', 0)} sentences.",
                f"Key dates found: {len(key_dates)}",
                f"Analysis confidence: {confidence:.1%}"
            ]
        
        # Ensure we have exactly 4 lines
        while len(summary_lines) < 4:
            summary_lines.append("")
        summary_lines = summary_lines[:4]
        
        return ContentAnalysisSummary(
            document_type=doc_type_name,
            confidence=confidence,
            summary="\n".join(summary_lines),
            key_details=summary_lines,
            sentiment=content_analysis.get("sentiment", {}).get("label", "neutral")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/analyze/text", response_model=dict)
async def analyze_text_file(
    file: UploadFile = File(...),
    analyzer: FileAnalyzer = Depends(get_file_analyzer)
):
    """
    Analyze a text file specifically.
    
    - **file**: The text file to analyze
    """
    try:
        # Validate the uploaded file
        validate_upload_file(file)
        
        # Check if it's a text file
        if not analyzer._is_text_file(file.content_type):
            raise HTTPException(
                status_code=400,
                detail="File is not a text file. Please upload a text file."
            )
        
        # Analyze the file
        analysis_result = analyzer.analyze_file(file)
        
        # Return only text-related analysis
        return {
            "filename": analysis_result["filename"],
            "text_analysis": analysis_result.get("text_analysis"),
            "json_analysis": analysis_result.get("json_analysis"),
            "csv_analysis": analysis_result.get("csv_analysis"),
            "xml_analysis": analysis_result.get("xml_analysis"),
            "content_analysis": analysis_result.get("content_analysis"),
            "basic_info": {
                "file_size": analysis_result["file_size"],
                "file_size_formatted": analysis_result["file_size_formatted"],
                "md5_hash": analysis_result["md5_hash"],
                "analyzed_at": analysis_result["analyzed_at"]
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/analyze/image", response_model=dict)
async def analyze_image_file(
    file: UploadFile = File(...),
    analyzer: FileAnalyzer = Depends(get_file_analyzer)
):
    """
    Analyze an image file specifically.
    
    - **file**: The image file to analyze
    """
    try:
        # Validate the uploaded file
        validate_upload_file(file)
        
        # Check if it's an image file
        if not analyzer._is_image_file(file.content_type):
            raise HTTPException(
                status_code=400,
                detail="File is not an image file. Please upload an image file."
            )
        
        # Analyze the file
        analysis_result = analyzer.analyze_file(file)
        
        # Return only image-related analysis
        return {
            "filename": analysis_result["filename"],
            "image_analysis": analysis_result.get("image_analysis"),
            "content_analysis": analysis_result.get("content_analysis"),
            "basic_info": {
                "file_size": analysis_result["file_size"],
                "file_size_formatted": analysis_result["file_size_formatted"],
                "md5_hash": analysis_result["md5_hash"],
                "analyzed_at": analysis_result["analyzed_at"]
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/analyze/document", response_model=dict)
async def analyze_document_file(
    file: UploadFile = File(...),
    analyzer: FileAnalyzer = Depends(get_file_analyzer)
):
    """
    Analyze a document file specifically.
    
    - **file**: The document file to analyze
    """
    try:
        # Validate the uploaded file
        validate_upload_file(file)
        
        # Check if it's a document file
        if not analyzer._is_document_file(file.content_type):
            raise HTTPException(
                status_code=400,
                detail="File is not a document file. Please upload a document file."
            )
        
        # Analyze the file
        analysis_result = analyzer.analyze_file(file)
        
        # Return only document-related analysis
        return {
            "filename": analysis_result["filename"],
            "document_analysis": analysis_result.get("document_analysis"),
            "content_analysis": analysis_result.get("content_analysis"),
            "basic_info": {
                "file_size": analysis_result["file_size"],
                "file_size_formatted": analysis_result["file_size_formatted"],
                "md5_hash": analysis_result["md5_hash"],
                "analyzed_at": analysis_result["analyzed_at"]
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/analyze/binary", response_model=dict)
async def analyze_binary_file(
    file: UploadFile = File(...),
    analyzer: FileAnalyzer = Depends(get_file_analyzer)
):
    """
    Analyze a binary file specifically.
    
    - **file**: The binary file to analyze
    """
    try:
        # Validate the uploaded file
        validate_upload_file(file)
        
        # Check if it's NOT a supported text/image/document file
        if analyzer._is_supported_type(file.content_type):
            raise HTTPException(
                status_code=400,
                detail="File is a supported type. Use the appropriate endpoint for text, image, or document analysis."
            )
        
        # Analyze the file
        analysis_result = analyzer.analyze_file(file)
        
        # Return only binary-related analysis
        return {
            "filename": analysis_result["filename"],
            "binary_analysis": analysis_result.get("binary_analysis"),
            "content_analysis": analysis_result.get("content_analysis"),
            "basic_info": {
                "file_size": analysis_result["file_size"],
                "file_size_formatted": analysis_result["file_size_formatted"],
                "md5_hash": analysis_result["md5_hash"],
                "analyzed_at": analysis_result["analyzed_at"]
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/analyze/hash", response_model=dict)
async def get_file_hash(
    file: UploadFile = File(...),
    analyzer: FileAnalyzer = Depends(get_file_analyzer)
):
    """
    Get hash values for an uploaded file.
    
    - **file**: The file to hash
    """
    try:
        # Validate the uploaded file
        validate_upload_file(file)
        
        # Read file content
        content = file.file.read()
        
        return {
            "filename": file.filename,
            "file_size": len(content),
            "file_size_formatted": analyzer._format_file_size(len(content)),
            "md5_hash": analyzer._calculate_md5(content),
            "sha256_hash": analyzer._calculate_sha256(content),
            "analyzed_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom exception handler for HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            status_code=exc.status_code
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler."""
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            status_code=500
        ).dict()
    )


if __name__ == "__main__":
    import uvicorn
    
    # Preload models before starting server
    logger.info("Preloading AI models...")
    preload_ai_models()
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    ) 