"""PDF text extraction module with improved handling for edge cases."""
import re
import fitz  # PyMuPDF
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExtractedPaper:
    """Represents extracted content from a research paper."""
    title: str
    full_text: str
    sections: dict[str, str]
    metadata: dict
    page_count: int = 0
    char_count: int = 0
    has_math: bool = False
    has_code: bool = False
    warnings: list[str] = field(default_factory=list)


# Configuration
MAX_PAGES = 50  # Limit for very large PDFs
MAX_CHARS = 100000  # ~25k tokens
MATH_INDICATORS = ['∑', '∫', '∂', '∇', '∞', 'α', 'β', 'γ', 'θ', 'λ', 'σ', '≤', '≥', '∈', '∀', '∃']


def extract_text_from_pdf(pdf_path: str | Path) -> ExtractedPaper:
    """
    Extract text content from a PDF research paper.
    Handles large files, math-heavy content, and edge cases.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        ExtractedPaper with structured content and warnings
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Check file size
    file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
    warnings = []

    if file_size_mb > 10:
        warnings.append(f"Large file ({file_size_mb:.1f}MB) - processing may be slow")

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise ValueError(f"Failed to open PDF: {e}")

    page_count = len(doc)

    # Handle very large PDFs
    if page_count > MAX_PAGES:
        warnings.append(f"PDF has {page_count} pages - only processing first {MAX_PAGES}")
        pages_to_process = MAX_PAGES
    else:
        pages_to_process = page_count

    # Extract metadata
    metadata = doc.metadata or {}
    title = metadata.get("title", "")

    # If no title in metadata, try to extract from first page
    if not title:
        title = _extract_title_from_first_page(doc[0]) if page_count > 0 else pdf_path.stem

    if not title:
        title = pdf_path.stem

    # Extract text from pages
    full_text = ""
    for i in range(pages_to_process):
        try:
            page = doc[i]
            page_text = page.get_text()
            full_text += page_text + "\n"

            # Check for truncation
            if len(full_text) > MAX_CHARS:
                warnings.append(f"Text truncated at {MAX_CHARS} characters")
                full_text = full_text[:MAX_CHARS]
                break
        except Exception as e:
            warnings.append(f"Error reading page {i+1}: {e}")

    doc.close()

    # Detect content characteristics
    has_math = any(char in full_text for char in MATH_INDICATORS)
    has_code = bool(re.search(r'```|def\s+\w+\(|function\s+\w+\(|class\s+\w+', full_text))

    if has_math:
        warnings.append("Paper contains mathematical notation - some formulas may not be captured accurately")

    # Clean text
    full_text = _clean_text(full_text)

    # Parse sections
    sections = _parse_sections(full_text)

    # Check for implementation details
    implementation_keywords = ['algorithm', 'pseudocode', 'implementation', 'code', 'procedure', 'function']
    has_implementation = any(kw in full_text.lower() for kw in implementation_keywords)

    if not has_implementation:
        warnings.append("Paper may lack detailed implementation - generated code will include assumptions")

    return ExtractedPaper(
        title=title,
        full_text=full_text,
        sections=sections,
        metadata=metadata,
        page_count=page_count,
        char_count=len(full_text),
        has_math=has_math,
        has_code=has_code,
        warnings=warnings
    )


def _extract_title_from_first_page(page) -> str:
    """Try to extract title from first page using font size heuristics."""
    try:
        blocks = page.get_text("dict")["blocks"]
        title_candidates = []

        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        # Large font = likely title
                        if span["size"] > 14:
                            text = span["text"].strip()
                            if len(text) > 10 and len(text) < 200:
                                title_candidates.append((span["size"], text))

        if title_candidates:
            # Return the one with largest font
            title_candidates.sort(reverse=True, key=lambda x: x[0])
            return title_candidates[0][1]
    except:
        pass
    return ""


def _clean_text(text: str) -> str:
    """Clean extracted text."""
    # Remove excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)

    # Remove page numbers and headers/footers (common patterns)
    text = re.sub(r'\n\d+\n', '\n', text)

    # Remove hyphenation at line breaks
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)

    return text.strip()


def _parse_sections(text: str) -> dict[str, str]:
    """
    Parse common paper sections from text.
    Uses improved heuristics to identify section boundaries.
    """
    sections = {}
    current_section = "preamble"
    current_content = []

    # Common section headers in research papers (case-insensitive)
    section_patterns = [
        (r'^(?:\d+\.?\s*)?abstract\s*$', "abstract"),
        (r'^(?:\d+\.?\s*)?introduction\s*$', "introduction"),
        (r'^(?:\d+\.?\s*)?(?:related\s*work|background|preliminaries|literature\s*review)\s*$', "background"),
        (r'^(?:\d+\.?\s*)?(?:method(?:s|ology)?|approach|proposed\s*(?:method|approach|system)|our\s*(?:method|approach))\s*$', "methodology"),
        (r'^(?:\d+\.?\s*)?(?:algorithm(?:s)?|the\s*algorithm)\s*$', "algorithm"),
        (r'^(?:\d+\.?\s*)?(?:implementation|system\s*(?:design|architecture))\s*$', "implementation"),
        (r'^(?:\d+\.?\s*)?(?:experiment(?:s)?|evaluation|results|empirical\s*study)\s*$', "experiments"),
        (r'^(?:\d+\.?\s*)?discussion\s*$', "discussion"),
        (r'^(?:\d+\.?\s*)?(?:conclusion(?:s)?|summary|concluding\s*remarks)\s*$', "conclusion"),
        (r'^(?:\d+\.?\s*)?(?:reference(?:s)?|bibliography)\s*$', "references"),
        (r'^(?:\d+\.?\s*)?(?:appendix|appendices)\s*', "appendix"),
    ]

    for line in text.split("\n"):
        line_stripped = line.strip()
        line_lower = line_stripped.lower()

        # Skip empty lines
        if not line_stripped:
            current_content.append(line)
            continue

        # Check if this line is a section header
        matched = False
        for pattern, section_name in section_patterns:
            if re.match(pattern, line_lower):
                # Save previous section
                if current_content:
                    content = "\n".join(current_content).strip()
                    if content:
                        sections[current_section] = content
                current_section = section_name
                current_content = []
                matched = True
                break

        if not matched:
            current_content.append(line)

    # Save last section
    if current_content:
        content = "\n".join(current_content).strip()
        if content:
            sections[current_section] = content

    return sections


def extract_from_text_file(text_path: str | Path) -> ExtractedPaper:
    """
    Extract content from a plain text file.
    """
    text_path = Path(text_path)
    if not text_path.exists():
        raise FileNotFoundError(f"File not found: {text_path}")

    full_text = text_path.read_text(encoding="utf-8")

    # Truncate if too large
    warnings = []
    if len(full_text) > MAX_CHARS:
        warnings.append(f"Text truncated at {MAX_CHARS} characters")
        full_text = full_text[:MAX_CHARS]

    full_text = _clean_text(full_text)
    sections = _parse_sections(full_text)

    has_math = any(char in full_text for char in MATH_INDICATORS)
    has_code = bool(re.search(r'```|def\s+\w+\(|function\s+\w+\(|class\s+\w+', full_text))

    return ExtractedPaper(
        title=text_path.stem,
        full_text=full_text,
        sections=sections,
        metadata={"source": "text_file"},
        page_count=1,
        char_count=len(full_text),
        has_math=has_math,
        has_code=has_code,
        warnings=warnings
    )


def get_paper_summary(paper: ExtractedPaper) -> str:
    """Get a human-readable summary of extracted paper."""
    summary = f"""
Paper: {paper.title}
Pages: {paper.page_count}
Characters: {paper.char_count:,}
Sections found: {', '.join(paper.sections.keys())}
Has math: {'Yes' if paper.has_math else 'No'}
Has code snippets: {'Yes' if paper.has_code else 'No'}
"""
    if paper.warnings:
        summary += "\nWarnings:\n" + "\n".join(f"  - {w}" for w in paper.warnings)
    return summary.strip()
