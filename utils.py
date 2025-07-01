import os
from typing import List
from fastapi import HTTPException, UploadFile
from config import settings


def validate_file_extension(filename: str) -> bool:
    """
    Validate if the file extension is allowed.
    
    Args:
        filename: The filename to validate
        
    Returns:
        bool: True if extension is allowed, False otherwise
    """
    if not filename:
        return False
    
    file_extension = os.path.splitext(filename)[1].lower()
    return file_extension in settings.allowed_extensions


def validate_file_size(file_size: int) -> bool:
    """
    Validate if the file size is within allowed limits.
    
    Args:
        file_size: The size of the file in bytes
        
    Returns:
        bool: True if size is within limits, False otherwise
    """
    return file_size <= settings.max_file_size


def validate_upload_file(file: UploadFile) -> None:
    """
    Validate uploaded file for extension and size.
    
    Args:
        file: The uploaded file to validate
        
    Raises:
        HTTPException: If validation fails
    """
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided"
        )
    
    if not validate_file_extension(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"File type not allowed. Allowed types: {', '.join(settings.allowed_extensions)}"
        )
    
    # Check file size if available
    if hasattr(file, 'size') and file.size:
        if not validate_file_size(file.size):
            max_size_mb = settings.max_file_size // (1024 * 1024)
            raise HTTPException(
                status_code=400,
                detail=f"File size too large. Maximum size allowed: {max_size_mb}MB"
            )


def get_file_size_mb(size_bytes: int) -> float:
    """
    Convert bytes to megabytes.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        float: Size in megabytes
    """
    return round(size_bytes / (1024 * 1024), 2)


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        str: Formatted file size
    """
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f}{size_names[i]}"


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing potentially dangerous characters.
    
    Args:
        filename: The filename to sanitize
        
    Returns:
        str: Sanitized filename
    """
    # Remove or replace potentially dangerous characters
    dangerous_chars = ['<', '>', ':', '"', '|', '?', '*', '\\', '/']
    sanitized = filename
    for char in dangerous_chars:
        sanitized = sanitized.replace(char, '_')
    
    return sanitized


def get_content_type_from_extension(filename: str) -> str:
    """
    Get content type based on file extension.
    
    Args:
        filename: The filename to check
        
    Returns:
        str: Content type
    """
    extension = os.path.splitext(filename)[1].lower()
    
    content_types = {
        '.pdf': 'application/pdf',
        '.doc': 'application/msword',
        '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        '.txt': 'text/plain',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.bmp': 'image/bmp',
        '.tiff': 'image/tiff',
        '.csv': 'text/csv',
        '.json': 'application/json',
        '.xml': 'application/xml',
        '.zip': 'application/zip',
        '.rar': 'application/x-rar-compressed'
    }
    
    return content_types.get(extension, 'application/octet-stream') 