"""
Sandbox utilities for safe execution of generated code.
Provides timeouts, resource limits, and isolation.
"""
import os
import sys
import signal
import subprocess
import threading
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class SandboxConfig:
    """Configuration for sandboxed execution."""
    timeout_seconds: int = 60
    max_output_bytes: int = 1024 * 1024  # 1MB
    allow_network: bool = True  # For local server
    working_dir: Optional[Path] = None


@dataclass
class SandboxResult:
    """Result of sandboxed execution."""
    success: bool
    return_code: int
    stdout: str
    stderr: str
    timed_out: bool = False
    error: Optional[str] = None


class Sandbox:
    """
    Provides sandboxed execution of generated code.

    Current implementation uses subprocess with timeouts.
    Future: Could use Docker containers for stronger isolation.
    """

    def __init__(self, config: Optional[SandboxConfig] = None):
        self.config = config or SandboxConfig()

    def run_python(
        self,
        script_path: Path,
        args: list[str] = None,
        env: dict = None
    ) -> SandboxResult:
        """Run a Python script in sandbox."""
        cmd = [sys.executable, str(script_path)]
        if args:
            cmd.extend(args)
        return self._run_command(cmd, env)

    def run_node(
        self,
        script_path: Path,
        args: list[str] = None,
        env: dict = None
    ) -> SandboxResult:
        """Run a Node.js script in sandbox."""
        cmd = ["node", str(script_path)]
        if args:
            cmd.extend(args)
        return self._run_command(cmd, env, shell=True)

    def check_syntax_python(self, script_path: Path) -> SandboxResult:
        """Check Python syntax without running."""
        cmd = [sys.executable, "-m", "py_compile", str(script_path)]
        return self._run_command(cmd, timeout=10)

    def check_syntax_node(self, script_path: Path) -> SandboxResult:
        """Check Node.js syntax without running."""
        cmd = ["node", "--check", str(script_path)]
        return self._run_command(cmd, timeout=10, shell=True)

    def _run_command(
        self,
        cmd: list[str],
        env: dict = None,
        timeout: int = None,
        shell: bool = False
    ) -> SandboxResult:
        """
        Run a command with safety limits.
        """
        timeout = timeout or self.config.timeout_seconds

        # Prepare environment
        run_env = os.environ.copy()
        if env:
            run_env.update(env)

        # Remove potentially dangerous env vars
        dangerous_vars = ['LD_PRELOAD', 'LD_LIBRARY_PATH', 'PYTHONPATH']
        for var in dangerous_vars:
            run_env.pop(var, None)

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=run_env,
                cwd=self.config.working_dir,
                shell=shell,
                # On Unix, create new process group for clean termination
                preexec_fn=os.setsid if sys.platform != 'win32' else None
            )

            try:
                stdout, stderr = process.communicate(timeout=timeout)

                # Truncate output if too large
                if len(stdout) > self.config.max_output_bytes:
                    stdout = stdout[:self.config.max_output_bytes] + b"\n[OUTPUT TRUNCATED]"
                if len(stderr) > self.config.max_output_bytes:
                    stderr = stderr[:self.config.max_output_bytes] + b"\n[OUTPUT TRUNCATED]"

                return SandboxResult(
                    success=process.returncode == 0,
                    return_code=process.returncode,
                    stdout=stdout.decode('utf-8', errors='replace'),
                    stderr=stderr.decode('utf-8', errors='replace'),
                    timed_out=False
                )

            except subprocess.TimeoutExpired:
                # Kill the process group
                self._kill_process(process)

                return SandboxResult(
                    success=False,
                    return_code=-1,
                    stdout="",
                    stderr="",
                    timed_out=True,
                    error=f"Process timed out after {timeout} seconds"
                )

        except FileNotFoundError as e:
            return SandboxResult(
                success=False,
                return_code=-1,
                stdout="",
                stderr="",
                error=f"Command not found: {e}"
            )
        except Exception as e:
            return SandboxResult(
                success=False,
                return_code=-1,
                stdout="",
                stderr="",
                error=f"Execution error: {e}"
            )

    def _kill_process(self, process: subprocess.Popen):
        """Kill a process and all its children."""
        try:
            if sys.platform == 'win32':
                # Windows: use taskkill
                subprocess.run(
                    ['taskkill', '/F', '/T', '/PID', str(process.pid)],
                    capture_output=True
                )
            else:
                # Unix: kill process group
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except:
            try:
                process.kill()
            except:
                pass


class DockerSandbox:
    """
    Docker-based sandbox for stronger isolation.
    Use when available for better security.
    """

    def __init__(self, image: str = "python:3.11-slim"):
        self.image = image
        self._docker_available = self._check_docker()

    def _check_docker(self) -> bool:
        """Check if Docker is available."""
        try:
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except:
            return False

    @property
    def available(self) -> bool:
        return self._docker_available

    def run_in_container(
        self,
        project_dir: Path,
        command: str,
        timeout: int = 60,
        memory_limit: str = "512m",
        cpu_limit: float = 1.0
    ) -> SandboxResult:
        """
        Run command in a Docker container with resource limits.
        """
        if not self._docker_available:
            return SandboxResult(
                success=False,
                return_code=-1,
                stdout="",
                stderr="",
                error="Docker not available"
            )

        docker_cmd = [
            "docker", "run",
            "--rm",  # Remove container after
            "--memory", memory_limit,
            "--cpus", str(cpu_limit),
            "--network", "host",  # Allow localhost access
            "-v", f"{project_dir}:/app",
            "-w", "/app",
            self.image,
            "sh", "-c", command
        ]

        try:
            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                timeout=timeout
            )

            return SandboxResult(
                success=result.returncode == 0,
                return_code=result.returncode,
                stdout=result.stdout.decode('utf-8', errors='replace'),
                stderr=result.stderr.decode('utf-8', errors='replace')
            )

        except subprocess.TimeoutExpired:
            return SandboxResult(
                success=False,
                return_code=-1,
                stdout="",
                stderr="",
                timed_out=True,
                error=f"Container timed out after {timeout} seconds"
            )
        except Exception as e:
            return SandboxResult(
                success=False,
                return_code=-1,
                stdout="",
                stderr="",
                error=f"Docker error: {e}"
            )


def get_sandbox(prefer_docker: bool = False) -> Sandbox:
    """
    Get appropriate sandbox based on environment.
    """
    if prefer_docker:
        docker = DockerSandbox()
        if docker.available:
            return docker
    return Sandbox()
