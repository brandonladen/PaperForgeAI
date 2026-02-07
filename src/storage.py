"""JSON-based storage for project history and analysis cache."""
import json
from pathlib import Path
from datetime import datetime
from dataclasses import asdict
from typing import Optional
from .analyzer import PaperAnalysis


class ProjectStorage:
    """Manages project history and cached analyses."""

    def __init__(self, storage_dir: str | Path = "storage"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.storage_dir / "history.json"
        self.analyses_dir = self.storage_dir / "analyses"
        self.analyses_dir.mkdir(exist_ok=True)

    def _load_history(self) -> list[dict]:
        """Load project history from JSON."""
        if self.history_file.exists():
            with open(self.history_file, encoding="utf-8") as f:
                return json.load(f)
        return []

    def _save_history(self, history: list[dict]):
        """Save project history to JSON."""
        with open(self.history_file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, default=str)

    def add_project(
        self,
        paper_path: str,
        analysis: PaperAnalysis,
        output_dir: str,
        status: str = "completed"
    ) -> str:
        """
        Record a generated project in history.

        Returns:
            Project ID
        """
        history = self._load_history()

        project_id = f"proj_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        entry = {
            "id": project_id,
            "paper_path": str(paper_path),
            "paper_title": analysis.title,
            "output_dir": str(output_dir),
            "status": status,
            "created_at": datetime.now().isoformat(),
            "algorithm": analysis.core_algorithm
        }

        history.append(entry)
        self._save_history(history)

        # Save full analysis
        self.save_analysis(project_id, analysis)

        return project_id

    def save_analysis(self, project_id: str, analysis: PaperAnalysis):
        """Save full analysis to separate file."""
        analysis_file = self.analyses_dir / f"{project_id}.json"
        with open(analysis_file, "w", encoding="utf-8") as f:
            json.dump(asdict(analysis), f, indent=2)

    def load_analysis(self, project_id: str) -> Optional[PaperAnalysis]:
        """Load analysis from cache."""
        analysis_file = self.analyses_dir / f"{project_id}.json"
        if not analysis_file.exists():
            return None
        with open(analysis_file, encoding="utf-8") as f:
            data = json.load(f)
            return PaperAnalysis(**data)

    def get_history(self, limit: int = 10) -> list[dict]:
        """Get recent project history."""
        history = self._load_history()
        return sorted(history, key=lambda x: x["created_at"], reverse=True)[:limit]

    def get_project(self, project_id: str) -> Optional[dict]:
        """Get a specific project by ID."""
        history = self._load_history()
        for entry in history:
            if entry["id"] == project_id:
                return entry
        return None

    def update_status(self, project_id: str, status: str):
        """Update project status."""
        history = self._load_history()
        for entry in history:
            if entry["id"] == project_id:
                entry["status"] = status
                entry["updated_at"] = datetime.now().isoformat()
                break
        self._save_history(history)

    def search_by_title(self, query: str) -> list[dict]:
        """Search projects by title."""
        history = self._load_history()
        query_lower = query.lower()
        return [
            entry for entry in history
            if query_lower in entry.get("paper_title", "").lower()
        ]

    def get_stats(self) -> dict:
        """Get storage statistics."""
        history = self._load_history()
        return {
            "total_projects": len(history),
            "completed": sum(1 for h in history if h.get("status") == "completed"),
            "failed": sum(1 for h in history if h.get("status") == "failed"),
            "storage_size_kb": sum(
                f.stat().st_size for f in self.storage_dir.rglob("*.json")
            ) / 1024
        }
