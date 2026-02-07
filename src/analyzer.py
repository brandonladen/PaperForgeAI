"""Multi-provider paper analysis module - supports OpenAI and Gemini."""
import json
from dataclasses import dataclass
from typing import Literal
from .pdf_parser import ExtractedPaper
from .config import get_openai_model, get_gemini_model


@dataclass
class PaperAnalysis:
    """Structured analysis of a research paper."""
    title: str
    summary: str
    core_algorithm: str
    algorithm_steps: list[str]
    data_structures: list[dict]
    inputs: list[dict]
    outputs: list[dict]
    dependencies: list[str]
    pseudocode: str
    implementation_notes: str


ANALYSIS_SYSTEM_PROMPT = """You are an expert software engineer and research paper analyst.
Your task is to analyze algorithm/systems research papers and extract implementation details.

You must respond with a valid JSON object containing the following fields:
- title: The paper's title or main topic
- summary: A 2-3 sentence summary of what the paper does
- core_algorithm: Name/description of the main algorithm or system
- algorithm_steps: Array of strings describing the step-by-step algorithm
- data_structures: Array of objects with {name, type, description} for each data structure needed
- inputs: Array of objects with {name, type, description} for expected inputs
- outputs: Array of objects with {name, type, description} for expected outputs
- dependencies: Array of Python package names needed (e.g., ["numpy", "scipy"])
- pseudocode: The core algorithm in pseudocode format
- implementation_notes: Any important implementation details or gotchas

Focus on making this implementable. Extract concrete, actionable information."""


ANALYSIS_USER_PROMPT = """Analyze this research paper and extract implementation details.

Paper Title: {title}

Paper Content:
{content}

Provide a JSON response with implementation details. Focus on:
1. The core algorithm/method
2. Data structures needed
3. Clear step-by-step process
4. What inputs it takes and outputs it produces

Respond ONLY with valid JSON, no markdown formatting."""


class PaperAnalyzer:
    """Analyzes research papers using OpenAI or Gemini."""

    def __init__(
        self,
        api_key: str,
        model: str = None,
        provider: Literal["openai", "gemini"] = "openai"
    ):
        """
        Initialize the analyzer.

        Args:
            api_key: API key for the provider
            model: Model to use (defaults from .env: OPENAI_MODEL or GEMINI_MODEL)
            provider: Which AI provider to use ("openai" or "gemini")
        """
        self.provider = provider
        self.api_key = api_key

        if provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
            self.model = model if model else get_openai_model()
        elif provider == "gemini":
            from google import genai
            self.client = genai.Client(api_key=api_key)
            self.model = model if model else get_gemini_model()
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def analyze(self, paper: ExtractedPaper, max_tokens: int = 4000) -> PaperAnalysis:
        """
        Analyze a paper and extract implementation details.

        Args:
            paper: Extracted paper content
            max_tokens: Maximum content tokens to send (truncates if needed)

        Returns:
            PaperAnalysis with structured information
        """
        # Prepare content - prioritize methodology and algorithm sections
        content = self._prepare_content(paper, max_tokens)

        # Call the appropriate provider
        if self.provider == "openai":
            result = self._call_openai(paper.title, content)
        else:
            result = self._call_gemini(paper.title, content)

        return PaperAnalysis(
            title=result.get("title", paper.title),
            summary=result.get("summary", ""),
            core_algorithm=result.get("core_algorithm", ""),
            algorithm_steps=result.get("algorithm_steps", []),
            data_structures=result.get("data_structures", []),
            inputs=result.get("inputs", []),
            outputs=result.get("outputs", []),
            dependencies=result.get("dependencies", ["typing"]),
            pseudocode=result.get("pseudocode", ""),
            implementation_notes=result.get("implementation_notes", "")
        )

    def _call_openai(self, title: str, content: str) -> dict:
        """Call OpenAI API."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
                {"role": "user", "content": ANALYSIS_USER_PROMPT.format(
                    title=title,
                    content=content
                )}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )

        result_text = response.choices[0].message.content
        return json.loads(result_text)

    def _call_gemini(self, title: str, content: str) -> dict:
        """Call Gemini API using new google-genai package."""
        from google import genai

        prompt = f"""{ANALYSIS_SYSTEM_PROMPT}

{ANALYSIS_USER_PROMPT.format(title=title, content=content)}

IMPORTANT: Respond with ONLY valid JSON, no markdown code blocks, no extra text."""

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                temperature=0.3,
                response_mime_type="application/json"
            )
        )

        result_text = response.text

        # Clean up response if it has markdown code blocks
        if result_text.startswith("```"):
            lines = result_text.split("\n")
            # Remove first and last lines if they're code block markers
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            result_text = "\n".join(lines)

        return json.loads(result_text)

    def _prepare_content(self, paper: ExtractedPaper, max_chars: int) -> str:
        """
        Prepare paper content for analysis, prioritizing important sections.
        """
        priority_sections = [
            "abstract",
            "methodology",
            "algorithm",
            "implementation",
            "introduction",
            "background",
            "experiments"
        ]

        content_parts = []
        char_count = 0

        # Add sections in priority order
        for section in priority_sections:
            if section in paper.sections and char_count < max_chars:
                section_text = paper.sections[section]
                remaining = max_chars - char_count
                if len(section_text) > remaining:
                    section_text = section_text[:remaining] + "..."
                content_parts.append(f"## {section.upper()}\n{section_text}")
                char_count += len(section_text)

        # If we have room and didn't get enough, add from full text
        if char_count < max_chars // 2:
            remaining = max_chars - char_count
            content_parts.append(f"\n## FULL TEXT\n{paper.full_text[:remaining]}")

        return "\n\n".join(content_parts)
