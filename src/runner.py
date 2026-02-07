"""Project runner - installs dependencies, validates, auto-fixes, and runs generated projects."""
import os
import sys
import subprocess
import threading
import time
import socket
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Callable, Literal

from .error_fixer import ErrorFixer, FixAttempt


@dataclass
class RunningProject:
    """Represents a running project."""
    project_id: str
    project_name: str
    project_dir: Path
    language: str
    port: int
    process: subprocess.Popen
    url: str
    status: str = "running"
    fix_attempts: list[FixAttempt] = field(default_factory=list)
    stdout_log: str = ""
    stderr_log: str = ""


class ProjectRunner:
    """Manages running generated projects with auto-fix capability."""

    MAX_PORT_ATTEMPTS = 100
    SERVER_TIMEOUT = 30

    def __init__(self):
        self.running_projects: dict[str, RunningProject] = {}
        self.base_port = 5001
        self._port_lock = threading.Lock()

    def _find_free_port(self) -> int:
        """Find an available port with retry logic."""
        with self._port_lock:
            port = self.base_port
            attempts = 0
            while attempts < self.MAX_PORT_ATTEMPTS:
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.bind(('127.0.0.1', port))
                        self.base_port = port + 1
                        return port
                except OSError:
                    port += 1
                    attempts += 1
            raise RuntimeError(f"No free ports available in range {self.base_port}-{self.base_port + self.MAX_PORT_ATTEMPTS}")

    def install_and_run(
        self,
        project_id: str,
        project_dir: Path,
        language: str,
        api_key: str = None,
        provider: Literal["openai", "gemini"] = "openai",
        on_log: Optional[Callable[[str, str], None]] = None
    ) -> RunningProject:
        """
        Install dependencies, validate code, auto-fix if needed, and run the project.

        Args:
            project_id: Unique project identifier
            project_dir: Path to project directory
            language: 'python' or 'nodejs'
            api_key: API key for error fixing (OpenAI or Gemini)
            provider: AI provider ('openai' or 'gemini')
            on_log: Callback for logging (message, type)

        Returns:
            RunningProject with URL
        """
        def log(msg: str, msg_type: str = "info"):
            if on_log:
                on_log(msg, msg_type)

        port = self._find_free_port()
        log(f"Allocated port {port}", "info")

        # Step 1: Install dependencies
        log(f"Installing dependencies...", "info")
        install_success = self._install_dependencies(project_dir, language, log)

        if not install_success:
            log("Dependency installation had warnings, continuing...", "warning")

        # Step 2: Inject port into project
        self._inject_port(project_dir, language, port)

        # Step 3: Validate and auto-fix if API key provided
        if api_key:
            log("Validating generated code...", "info")
            fixer = ErrorFixer(api_key, provider=provider)
            success, _, fix_history = fixer.run_and_fix(
                project_dir, language, on_log
            )
            # Re-inject port after fixes (fixer may have regenerated files)
            if fix_history:
                self._inject_port(project_dir, language, port)

        # Step 4: Start the server
        log("Starting server...", "info")
        process, stdout_thread, stderr_thread = self._start_server(project_dir, language, port)

        # Step 5: Wait for server to be ready
        url = f"http://localhost:{port}"
        log(f"Waiting for server at {url}...", "info")

        server_ready = self._wait_for_server(port, timeout=self.SERVER_TIMEOUT)

        if server_ready:
            log(f"✓ Server is running!", "success")
            log(f"✓ Access your MVP at: {url}", "success")
        else:
            # Check if process died
            if process.poll() is not None:
                # Process exited - capture error
                stderr_output = self._read_process_output(process, stderr_thread)
                log(f"Server failed to start: {stderr_output[:300]}", "error")

                # Try auto-fix one more time if we have API key
                if api_key:
                    log("Attempting auto-fix...", "warning")
                    fixer = ErrorFixer(api_key, provider=provider)
                    success, _, _ = fixer.run_and_fix(project_dir, language, on_log)
                    if success:
                        # Re-inject port and restart
                        self._inject_port(project_dir, language, port)
                        process, stdout_thread, stderr_thread = self._start_server(project_dir, language, port)
                        server_ready = self._wait_for_server(port, timeout=self.SERVER_TIMEOUT)
                        if server_ready:
                            log(f"✓ Server recovered after fix!", "success")
                        else:
                            log(f"Server still failing. Manual review needed.", "error")
                    else:
                        log(f"Auto-fix could not resolve the error.", "error")
            else:
                log(f"Server may still be starting. Try: {url}", "warning")

        # Track running project
        running = RunningProject(
            project_id=project_id,
            project_name=project_dir.name,
            project_dir=project_dir,
            language=language,
            port=port,
            process=process,
            url=url,
            status="running" if server_ready else "starting"
        )
        self.running_projects[project_id] = running

        return running

    def _install_dependencies(self, project_dir: Path, language: str, log: Callable) -> bool:
        """Install project dependencies."""
        # Ensure absolute path for project_dir
        project_dir = Path(project_dir).resolve()

        try:
            if language == "python":
                req_file = project_dir / "requirements.txt"
                if req_file.exists():
                    result = subprocess.run(
                        [sys.executable, "-m", "pip", "install", "-r", str(req_file.resolve()), "-q", "--disable-pip-version-check"],
                        cwd=str(project_dir),
                        capture_output=True,
                        text=True,
                        timeout=120
                    )
                    if result.returncode == 0:
                        log("✓ Python dependencies installed", "success")
                        return True
                    else:
                        log(f"pip warning: {result.stderr[:150]}", "warning")
                        return False
            else:  # nodejs
                result = subprocess.run(
                    ["npm", "install", "--silent", "--no-audit", "--no-fund"],
                    cwd=str(project_dir),
                    capture_output=True,
                    text=True,
                    timeout=120,
                    shell=True
                )
                if result.returncode == 0:
                    log("✓ Node.js dependencies installed", "success")
                    return True
                else:
                    log(f"npm warning: {result.stderr[:150]}", "warning")
                    return False
        except subprocess.TimeoutExpired:
            log("Dependency installation timed out", "warning")
            return False
        except Exception as e:
            log(f"Dependency error: {str(e)[:100]}", "warning")
            return False

        return True

    def _inject_port(self, project_dir: Path, language: str, port: int):
        """Inject port number into the project's config or main file."""
        if language == "python":
            app_file = project_dir / "app.py"
            if app_file.exists():
                content = app_file.read_text(encoding="utf-8")
                # Update PORT in the file
                import re
                content = re.sub(
                    r"port\s*=\s*int\(os\.environ\.get\(['\"]PORT['\"],\s*\d+\)\)",
                    f"port = int(os.environ.get('PORT', {port}))",
                    content
                )
                app_file.write_text(content, encoding="utf-8")
        else:
            app_file = project_dir / "app.js"
            if app_file.exists():
                content = app_file.read_text(encoding="utf-8")
                import re
                content = re.sub(
                    r"PORT\s*\|\|\s*\d+",
                    f"PORT || {port}",
                    content
                )
                app_file.write_text(content, encoding="utf-8")

    def _start_server(self, project_dir: Path, language: str, port: int):
        """Start the server process with proper output capture."""
        env = os.environ.copy()
        env["PORT"] = str(port)

        if language == "python":
            process = subprocess.Popen(
                [sys.executable, "app.py"],
                cwd=project_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
        else:
            process = subprocess.Popen(
                ["node", "app.js"],
                cwd=project_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=True
            )

        # Start threads to capture output
        stdout_lines = []
        stderr_lines = []

        def read_stdout():
            for line in process.stdout:
                stdout_lines.append(line)

        def read_stderr():
            for line in process.stderr:
                stderr_lines.append(line)

        stdout_thread = threading.Thread(target=read_stdout, daemon=True)
        stderr_thread = threading.Thread(target=read_stderr, daemon=True)
        stdout_thread.start()
        stderr_thread.start()

        return process, stdout_thread, stderr_thread

    def _read_process_output(self, process, stderr_thread, timeout=2):
        """Read captured output from process."""
        stderr_thread.join(timeout=timeout)
        try:
            return process.stderr.read() if process.stderr else ""
        except:
            return ""

    def _wait_for_server(self, port: int, timeout: int = 30) -> bool:
        """Wait for server to be ready with health check."""
        import urllib.request
        import urllib.error

        start = time.time()
        while time.time() - start < timeout:
            try:
                # Try health endpoint first
                urllib.request.urlopen(f"http://localhost:{port}/health", timeout=2)
                return True
            except (urllib.error.URLError, urllib.error.HTTPError, ConnectionRefusedError, OSError):
                pass

            try:
                # Try root endpoint as fallback
                urllib.request.urlopen(f"http://localhost:{port}/", timeout=2)
                return True
            except (urllib.error.URLError, urllib.error.HTTPError, ConnectionRefusedError, OSError):
                pass

            time.sleep(0.5)
        return False

    def stop_project(self, project_id: str) -> bool:
        """Stop a running project gracefully."""
        if project_id not in self.running_projects:
            return False

        project = self.running_projects[project_id]
        try:
            project.process.terminate()
            project.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            project.process.kill()
        except:
            try:
                project.process.kill()
            except:
                pass

        project.status = "stopped"
        del self.running_projects[project_id]
        return True

    def stop_all(self):
        """Stop all running projects."""
        for project_id in list(self.running_projects.keys()):
            self.stop_project(project_id)

    def get_running(self) -> list[dict]:
        """Get list of running projects."""
        return [
            {
                "project_id": p.project_id,
                "project_name": p.project_name,
                "language": p.language,
                "port": p.port,
                "url": p.url,
                "status": p.status
            }
            for p in self.running_projects.values()
        ]

    def is_running(self, project_id: str) -> bool:
        """Check if a project is still running."""
        if project_id not in self.running_projects:
            return False
        project = self.running_projects[project_id]
        return project.process.poll() is None

    def get_logs(self, project_id: str) -> dict:
        """Get stdout/stderr logs for a project."""
        if project_id not in self.running_projects:
            return {"stdout": "", "stderr": "", "error": "Project not found"}
        project = self.running_projects[project_id]
        return {
            "stdout": project.stdout_log,
            "stderr": project.stderr_log
        }


# Global runner instance
runner = ProjectRunner()
