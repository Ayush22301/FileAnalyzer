import os
import io
from typing import Dict, Any, List, Optional
from fastapi import HTTPException, UploadFile
import mimetypes
import hashlib
from datetime import datetime
import json
from content_analyzer import ContentAnalyzer


class FileAnalyzer:
    """Service class for analyzing uploaded files without storing them."""
    
    def __init__(self):
        """Initialize the file analyzer."""
        self.supported_text_types = {
            'text/plain', 'text/csv', 'application/json', 'application/xml',
            'text/html', 'text/css', 'text/javascript', 'application/javascript'
        }
        
        self.supported_image_types = {
            'image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/bmp',
            'image/tiff', 'image/webp'
        }
        
        self.supported_document_types = {
            'application/pdf', 'application/msword',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'application/vnd.ms-excel',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        }
        
        # Initialize content analyzer
        self.content_analyzer = ContentAnalyzer()
    
    def analyze_file(self, file: UploadFile) -> Dict[str, Any]:
        """
        Analyze an uploaded file and return comprehensive information.
        
        Args:
            file: The uploaded file to analyze
            
        Returns:
            dict: Analysis results
        """
        try:
            # Read file content
            content = file.file.read()
            file.file.seek(0)  # Reset file pointer for potential future reads
            
            # Basic file information
            analysis = {
                "filename": file.filename,
                "content_type": file.content_type,
                "file_size": len(content),
                "file_size_formatted": self._format_file_size(len(content)),
                "file_extension": self._get_file_extension(file.filename),
                "md5_hash": self._calculate_md5(content),
                "sha256_hash": self._calculate_sha256(content),
                "analyzed_at": datetime.utcnow().isoformat(),
                "analysis_type": self._determine_analysis_type(file.content_type),
                "is_text": self._is_text_file(file.content_type),
                "is_image": self._is_image_file(file.content_type),
                "is_document": self._is_document_file(file.content_type),
                "is_supported": self._is_supported_type(file.content_type)
            }
            
            # Perform specific analysis based on file type
            if analysis["is_text"]:
                analysis.update(self._analyze_text_content(content, file.content_type))
            elif analysis["is_image"]:
                analysis.update(self._analyze_image_content(content, file.content_type))
            elif analysis["is_document"]:
                analysis.update(self._analyze_document_content(content, file.content_type))
            else:
                analysis.update(self._analyze_binary_content(content))
            
            # Add metadata analysis
            analysis["metadata"] = self._extract_metadata(file, content)
            
            # Add intelligent content analysis
            content_analysis = self.content_analyzer.analyze_content(content, file.content_type, file.filename)
            analysis.update(content_analysis)
            
            return analysis
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error analyzing file: {str(e)}"
            )
    
    def _get_file_extension(self, filename: str) -> str:
        """Get file extension from filename."""
        if not filename:
            return ""
        return os.path.splitext(filename)[1].lower()
    
    def _calculate_md5(self, content: bytes) -> str:
        """Calculate MD5 hash of file content."""
        return hashlib.md5(content).hexdigest()
    
    def _calculate_sha256(self, content: bytes) -> str:
        """Calculate SHA256 hash of file content."""
        return hashlib.sha256(content).hexdigest()
    
    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format."""
        if size_bytes == 0:
            return "0B"
        
        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        
        return f"{size_bytes:.1f}{size_names[i]}"
    
    def _determine_analysis_type(self, content_type: str) -> str:
        """Determine the type of analysis to perform."""
        if self._is_text_file(content_type):
            return "text"
        elif self._is_image_file(content_type):
            return "image"
        elif self._is_document_file(content_type):
            return "document"
        else:
            return "binary"
    
    def _is_text_file(self, content_type: str) -> bool:
        """Check if file is a text file."""
        return content_type in self.supported_text_types or content_type.startswith('text/')
    
    def _is_image_file(self, content_type: str) -> bool:
        """Check if file is an image file."""
        return content_type in self.supported_image_types or content_type.startswith('image/')
    
    def _is_document_file(self, content_type: str) -> bool:
        """Check if file is a document file."""
        return content_type in self.supported_document_types
    
    def _is_supported_type(self, content_type: str) -> bool:
        """Check if file type is supported for analysis."""
        return (self._is_text_file(content_type) or 
                self._is_image_file(content_type) or 
                self._is_document_file(content_type))
    
    def _analyze_text_content(self, content: bytes, content_type: str) -> Dict[str, Any]:
        """Analyze text file content."""
        try:
            text_content = content.decode('utf-8')
            
            analysis = {
                "text_analysis": {
                    "character_count": len(text_content),
                    "word_count": len(text_content.split()),
                    "line_count": len(text_content.splitlines()),
                    "paragraph_count": len([p for p in text_content.split('\n\n') if p.strip()]),
                    "average_word_length": self._calculate_average_word_length(text_content),
                    "unique_words": len(set(text_content.lower().split())),
                    "encoding": "utf-8"
                }
            }
            
            # Specific analysis based on content type
            if content_type == 'application/json':
                analysis["json_analysis"] = self._analyze_json_content(text_content)
            elif content_type == 'text/csv':
                analysis["csv_analysis"] = self._analyze_csv_content(text_content)
            elif content_type == 'application/xml':
                analysis["xml_analysis"] = self._analyze_xml_content(text_content)
            
            return analysis
            
        except UnicodeDecodeError:
            return {
                "text_analysis": {
                    "error": "Unable to decode text content",
                    "encoding": "unknown"
                }
            }
    
    def _analyze_image_content(self, content: bytes, content_type: str) -> Dict[str, Any]:
        """Analyze image file content."""
        try:
            from PIL import Image
            import io
            
            image = Image.open(io.BytesIO(content))
            
            return {
                "image_analysis": {
                    "format": image.format,
                    "mode": image.mode,
                    "width": image.width,
                    "height": image.height,
                    "aspect_ratio": round(image.width / image.height, 2),
                    "color_depth": self._get_color_depth(image.mode),
                    "file_size": len(content),
                    "compression": self._estimate_compression(content, image.width, image.height, image.mode)
                }
            }
            
        except ImportError:
            return {
                "image_analysis": {
                    "error": "PIL/Pillow not installed for image analysis",
                    "file_size": len(content)
                }
            }
        except Exception as e:
            return {
                "image_analysis": {
                    "error": f"Error analyzing image: {str(e)}",
                    "file_size": len(content)
                }
            }
    
    def _analyze_document_content(self, content: bytes, content_type: str) -> Dict[str, Any]:
        """Analyze document file content."""
        return {
            "document_analysis": {
                "content_type": content_type,
                "file_size": len(content),
                "note": "Document content analysis available through content_analysis field"
            }
        }
    
    def _analyze_binary_content(self, content: bytes) -> Dict[str, Any]:
        """Analyze binary file content."""
        return {
            "binary_analysis": {
                "file_size": len(content),
                "entropy": self._calculate_entropy(content),
                "null_bytes": content.count(b'\x00'),
                "printable_ratio": self._calculate_printable_ratio(content)
            }
        }
    
    def _extract_metadata(self, file: UploadFile, content: bytes) -> Dict[str, Any]:
        """Extract metadata from file."""
        return {
            "original_filename": file.filename,
            "content_type": file.content_type,
            "content_length": len(content),
            "upload_timestamp": datetime.utcnow().isoformat()
        }
    
    def _calculate_average_word_length(self, text: str) -> float:
        """Calculate average word length in text."""
        words = text.split()
        if not words:
            return 0.0
        return round(sum(len(word) for word in words) / len(words), 2)
    
    def _analyze_json_content(self, text_content: str) -> Dict[str, Any]:
        """Analyze JSON content."""
        try:
            data = json.loads(text_content)
            return {
                "is_valid": True,
                "structure_type": type(data).__name__,
                "size": len(data) if isinstance(data, (list, dict)) else 1,
                "depth": self._calculate_json_depth(data)
            }
        except json.JSONDecodeError as e:
            return {
                "is_valid": False,
                "error": str(e)
            }
    
    def _analyze_csv_content(self, text_content: str) -> Dict[str, Any]:
        """Analyze CSV content."""
        lines = text_content.splitlines()
        if not lines:
            return {"rows": 0, "columns": 0}
        
        # Estimate columns from first line
        first_line = lines[0]
        columns = len(first_line.split(','))
        
        return {
            "rows": len(lines),
            "columns": columns,
            "estimated_size": len(lines) * columns
        }
    
    def _analyze_xml_content(self, text_content: str) -> Dict[str, Any]:
        """Analyze XML content."""
        try:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(text_content)
            return {
                "is_valid": True,
                "root_tag": root.tag,
                "total_elements": len(list(root.iter())),
                "depth": self._calculate_xml_depth(root)
            }
        except ET.ParseError as e:
            return {
                "is_valid": False,
                "error": str(e)
            }
    
    def _calculate_json_depth(self, obj, current_depth=0):
        """Calculate the maximum depth of a JSON object."""
        if isinstance(obj, dict):
            if not obj:
                return current_depth
            return max(self._calculate_json_depth(v, current_depth + 1) for v in obj.values())
        elif isinstance(obj, list):
            if not obj:
                return current_depth
            return max(self._calculate_json_depth(item, current_depth + 1) for item in obj)
        else:
            return current_depth
    
    def _calculate_xml_depth(self, element, current_depth=0):
        """Calculate the maximum depth of an XML element."""
        if not list(element):
            return current_depth
        return max(self._calculate_xml_depth(child, current_depth + 1) for child in element)
    
    def _get_color_depth(self, mode: str) -> str:
        """Get color depth from PIL image mode."""
        depth_map = {
            'L': '8-bit grayscale',
            'RGB': '24-bit color',
            'RGBA': '32-bit color',
            'CMYK': '32-bit color',
            'P': '8-bit palette',
            '1': '1-bit binary'
        }
        return depth_map.get(mode, 'unknown')
    
    def _estimate_compression(self, content: bytes, width: int, height: int, mode: str) -> str:
        """Estimate image compression."""
        # Rough estimation based on file size vs expected size
        if mode == 'RGB':
            expected_size = width * height * 3
        elif mode == 'RGBA':
            expected_size = width * height * 4
        elif mode == 'L':
            expected_size = width * height
        else:
            return "unknown"
        
        actual_size = len(content)
        compression_ratio = actual_size / expected_size if expected_size > 0 else 0
        
        if compression_ratio < 0.1:
            return "highly compressed"
        elif compression_ratio < 0.5:
            return "compressed"
        elif compression_ratio < 0.8:
            return "lightly compressed"
        else:
            return "uncompressed"
    
    def _calculate_entropy(self, data: bytes) -> float:
        """Calculate Shannon entropy of binary data."""
        if not data:
            return 0.0
        
        # Count byte frequencies
        byte_counts = [0] * 256
        for byte in data:
            byte_counts[byte] += 1
        
        # Calculate entropy
        entropy = 0.0
        data_length = len(data)
        
        for count in byte_counts:
            if count > 0:
                probability = count / data_length
                entropy -= probability * (probability.bit_length() - 1)
        
        return round(entropy, 2)
    
    def _calculate_printable_ratio(self, data: bytes) -> float:
        """Calculate ratio of printable ASCII characters."""
        if not data:
            return 0.0
        
        printable_count = sum(1 for byte in data if 32 <= byte <= 126)
        return round(printable_count / len(data), 3) 