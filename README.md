<<<<<<< HEAD
# FileAnalyzer
=======
# Document Analyzer API

A FastAPI backend for intelligent file analysis without storing files. Analyze files in memory and get detailed insights about their content, structure, and meaning.

## Features

- 🧠 **Intelligent Content Analysis**: Understand what documents are about
- 📊 **Document Type Detection**: Automatically identify bank statements, invoices, resumes, contracts, etc.
- 🔍 **Entity Extraction**: Extract names, dates, amounts, and key information
- 📝 **Content Summarization**: Generate meaningful summaries of document content
- 🎭 **Sentiment Analysis**: Analyze document tone and sentiment
- 🖼️ **Image Analysis**: Dimensions, format, color depth, compression estimation
- 📄 **Document Parsing**: Extract text from PDFs, Word docs, Excel files
- 🔢 **Binary Analysis**: Entropy calculation, null byte detection, printable ratio
- 🆔 **Hash Generation**: MD5 and SHA256 hashes for file integrity
- 📋 **Format Support**: JSON, CSV, XML, and various image formats
- ✅ **File Validation**: Validate file types and sizes
- 🏥 **Health Checks**: Monitor API health
- 📚 **Auto-generated Docs**: Interactive API documentation

## Prerequisites

- Python 3.8+
- No external storage required (files are analyzed in memory)

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd document_analyzer
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install additional models** (optional, for enhanced analysis):
   ```bash
   # Install spaCy model for NLP
   python -m spacy download en_core_web_sm
   
   # Install Tesseract for OCR (system dependent)
   # Ubuntu/Debian: sudo apt-get install tesseract-ocr
   # macOS: brew install tesseract
   # Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
   ```

5. **Set up environment variables** (optional):
   ```bash
   cp env.example .env
   ```
   
   Edit `.env` file if needed:
   ```env
   APP_NAME=Document Analyzer API
   DEBUG=True
   HOST=0.0.0.0
   PORT=8000
   MAX_FILE_SIZE_MB=10
   ```

## Running the Application

### Development Mode
```bash
python main.py
```

### Production Mode
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## API Documentation

Once the server is running, you can access:
- **Interactive API Docs**: http://localhost:8000/docs
- **ReDoc Documentation**: http://localhost:8000/redoc

## API Endpoints

### Health Check
- `GET /health` - Check API health status

### File Analysis
- `GET /supported-formats` - Get list of supported file formats
- `POST /analyze` - Comprehensive file analysis
- `POST /analyze/content` - **NEW**: Intelligent content analysis with document understanding
- `POST /analyze/text` - Text file analysis
- `POST /analyze/image` - Image file analysis
- `POST /analyze/document` - Document file analysis
- `POST /analyze/binary` - Binary file analysis
- `POST /analyze/hash` - Get file hash values

## Usage Examples

### Intelligent Content Analysis (NEW)
```bash
curl -X POST "http://localhost:8000/analyze/content" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@bank_statement.pdf"
```

**Response Example:**
```json
{
  "document_type": "Bank Statement",
  "confidence": 0.85,
  "summary": "This is a bank statement for John Doe showing account balance of $5,240.50 as of December 2024.",
  "key_details": [
    "Account Number: 1234567890",
    "Balance: $5,240.50",
    "Statement Period: December 2024",
    "Customer: John Doe",
    "Amount: $1,250.00"
  ],
  "sentiment": "neutral"
}
```

### Comprehensive File Analysis
```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.txt"
```

### Text File Analysis
```bash
curl -X POST "http://localhost:8000/analyze/text" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data.csv"
```

### Image Analysis
```bash
curl -X POST "http://localhost:8000/analyze/image" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.jpg"
```

### Get File Hash
```bash
curl -X POST "http://localhost:8000/analyze/hash" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@file.pdf"
```

### Get Supported Formats
```bash
curl -X GET "http://localhost:8000/supported-formats" \
  -H "accept: application/json"
```

## Analysis Results

### Text Analysis
```json
{
  "filename": "document.txt",
  "text_analysis": {
    "character_count": 1250,
    "word_count": 200,
    "line_count": 25,
    "paragraph_count": 5,
    "average_word_length": 6.25,
    "unique_words": 150,
    "encoding": "utf-8"
  },
  "basic_info": {
    "file_size": 1250,
    "file_size_formatted": "1.2KB",
    "md5_hash": "abc123...",
    "analyzed_at": "2024-01-01T12:00:00"
  }
}
```

### Image Analysis
```json
{
  "filename": "image.jpg",
  "image_analysis": {
    "format": "JPEG",
    "mode": "RGB",
    "width": 1920,
    "height": 1080,
    "aspect_ratio": 1.78,
    "color_depth": "24-bit color",
    "file_size": 250000,
    "compression": "compressed"
  },
  "basic_info": {
    "file_size": 250000,
    "file_size_formatted": "244.1KB",
    "md5_hash": "def456...",
    "analyzed_at": "2024-01-01T12:00:00"
  }
}
```

### Binary Analysis
```json
{
  "filename": "binary.exe",
  "binary_analysis": {
    "file_size": 1024000,
    "entropy": 7.85,
    "null_bytes": 150,
    "printable_ratio": 0.45
  },
  "basic_info": {
    "file_size": 1024000,
    "file_size_formatted": "1000.0KB",
    "md5_hash": "ghi789...",
    "analyzed_at": "2024-01-01T12:00:00"
  }
}
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `APP_NAME` | Application Name | `Document Analyzer API` |
| `DEBUG` | Debug Mode | `True` |
| `HOST` | Server Host | `0.0.0.0` |
| `PORT` | Server Port | `8000` |
| `MAX_FILE_SIZE_MB` | Maximum file size in MB | `10` |
| `ENABLE_IMAGE_ANALYSIS` | Enable image analysis | `True` |
| `ENABLE_TEXT_ANALYSIS` | Enable text analysis | `True` |
| `ENABLE_DOCUMENT_ANALYSIS` | Enable document analysis | `True` |
| `ENABLE_BINARY_ANALYSIS` | Enable binary analysis | `True` |

### Supported File Types

#### Text Files
- `.txt`, `.csv`, `.json`, `.xml`
- `.html`, `.css`, `.js`
- Any text-based format

#### Image Files
- `.jpg`, `.jpeg`, `.png`, `.gif`
- `.bmp`, `.tiff`, `.webp`

#### Document Files
- `.pdf`, `.doc`, `.docx`
- `.xls`, `.xlsx`

#### Binary Files
- Any other file type

## Project Structure

```
document_analyzer/
├── main.py              # FastAPI application and endpoints
├── config.py            # Configuration and settings
├── file_analyzer.py     # File analysis service
├── models.py            # Pydantic models
├── utils.py             # Utility functions
├── requirements.txt     # Python dependencies
├── env.example          # Environment variables template
└── README.md           # This file
```

## Error Handling

The API includes comprehensive error handling:
- File validation errors (type, size)
- Analysis errors for unsupported formats
- Memory errors for large files
- General server errors

All errors return structured JSON responses with appropriate HTTP status codes.

## Security Considerations

- Files are processed in memory and not stored
- File size limits prevent memory exhaustion
- File type validation prevents malicious uploads
- No persistent storage of uploaded files
- Input sanitization for all file operations

## Development

### Adding New Analysis Types

To add support for new file types, update the `FileAnalyzer` class in `file_analyzer.py`:

```python
def _analyze_custom_content(self, content: bytes, content_type: str) -> Dict[str, Any]:
    """Analyze custom file content."""
    return {
        "custom_analysis": {
            "custom_property": "value"
        }
    }
```

### Customizing File Size Limits

Update the `max_file_size` setting in `config.py`:

```python
max_file_size: int = 50 * 1024 * 1024  # 50MB
```

## Performance Notes

- Files are processed entirely in memory
- Large files may consume significant RAM
- Consider adjusting `max_file_size` based on server resources
- Image analysis requires PIL/Pillow library
- Binary analysis includes entropy calculation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License. 
>>>>>>> 2f86d8d (Initial commit)
