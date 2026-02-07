"""
Docker Deployment Module
Builds and runs generated MVP projects using Docker.
Supports local deployment and Docker Hub push.
"""
import subprocess
import time
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Callable


@dataclass
class DeploymentResult:
    """Result of a deployment operation."""
    success: bool
    url: Optional[str] = None
    container_id: Optional[str] = None
    container_name: Optional[str] = None
    image_name: Optional[str] = None
    error: Optional[str] = None
    logs: str = ""


@dataclass
class PushResult:
    """Result of a Docker Hub push operation."""
    success: bool
    image_url: Optional[str] = None
    full_tag: Optional[str] = None
    error: Optional[str] = None
    logs: str = ""


class DockerDeployer:
    """
    Deploy generated MVPs using Docker.
    Builds image, runs container, and returns access URL.
    """

    def __init__(self, default_port: int = 5001):
        self.default_port = default_port

    def check_docker(self) -> bool:
        """Check if Docker is installed and running."""
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def deploy(
        self,
        project_dir: Path,
        project_name: str,
        port: int = None,
        on_progress: Optional[Callable[[str, str], None]] = None
    ) -> DeploymentResult:
        """
        Build and run the project in Docker.

        Args:
            project_dir: Path to the generated project
            project_name: Name for the container/image
            port: Port to expose (default: 5001)
            on_progress: Progress callback

        Returns:
            DeploymentResult with URL and container info
        """
        def log(stage: str, msg: str):
            if on_progress:
                on_progress(stage, msg)

        port = port or self.default_port
        project_dir = Path(project_dir)

        # Sanitize project name for Docker
        container_name = self._sanitize_name(project_name)
        image_name = f"paperforge-{container_name}"

        logs = []

        # Check Docker is available
        log("deploy", "Checking Docker...")
        if not self.check_docker():
            return DeploymentResult(
                success=False,
                error="Docker is not installed or not running. Please start Docker and try again."
            )

        # Stop any existing container with same name or using the target port
        log("deploy", "Cleaning up old containers...")
        self._stop_container(container_name)
        self._free_port(port)  # Also free the port in case a different container is using it

        # Build the image
        log("deploy", f"Building Docker image: {image_name}...")
        build_result = self._build_image(project_dir, image_name)
        logs.append(f"BUILD:\n{build_result['output']}")

        if not build_result["success"]:
            return DeploymentResult(
                success=False,
                error=f"Docker build failed: {build_result['error']}",
                logs="\n".join(logs)
            )

        log("deploy", "Image built successfully.")

        # Run the container
        log("deploy", f"Starting container on port {port}...")
        run_result = self._run_container(
            image_name=image_name,
            container_name=container_name,
            port=port
        )
        logs.append(f"RUN:\n{run_result['output']}")

        if not run_result["success"]:
            return DeploymentResult(
                success=False,
                error=f"Failed to start container: {run_result['error']}",
                logs="\n".join(logs)
            )

        # Wait for container to be healthy
        log("deploy", "Waiting for application to start...")
        healthy = self._wait_for_healthy(container_name, port, timeout=30)

        if not healthy:
            # Get container logs for debugging
            container_logs = self._get_container_logs(container_name)
            logs.append(f"CONTAINER LOGS:\n{container_logs}")
            return DeploymentResult(
                success=False,
                container_id=run_result.get("container_id"),
                container_name=container_name,
                error="Container started but health check failed. Check logs for errors.",
                logs="\n".join(logs)
            )

        url = f"http://localhost:{port}"
        log("deploy", f"Deployment successful! URL: {url}")

        return DeploymentResult(
            success=True,
            url=url,
            container_id=run_result.get("container_id"),
            container_name=container_name,
            image_name=image_name,
            logs="\n".join(logs)
        )

    def push_to_dockerhub(
        self,
        image_name: str,
        username: str = None,
        password: str = None,
        tag: str = "latest",
        on_progress: Optional[Callable[[str, str], None]] = None
    ) -> PushResult:
        """
        Push a local Docker image to Docker Hub.

        Args:
            image_name: Local image name (e.g., "paperforge-myproject")
            username: Docker Hub username (optional if already logged in)
            password: Docker Hub password/token (optional if already logged in)
            tag: Tag for the image (default: "latest")
            on_progress: Progress callback

        Returns:
            PushResult with image URL and status

        Note: If username/password not provided, uses existing Docker credentials
              from previous `docker login` command.
        """
        def log(stage: str, msg: str):
            if on_progress:
                on_progress(stage, msg)

        logs = []

        # Check Docker is available
        log("push", "Checking Docker...")
        if not self.check_docker():
            return PushResult(
                success=False,
                error="Docker is not installed or not running"
            )

        # Get username from existing login if not provided
        if not username:
            username = self._get_logged_in_user()
            if not username:
                return PushResult(
                    success=False,
                    error="No Docker Hub username provided and not logged in. Run 'docker login' first or provide username."
                )
            log("push", f"Using existing Docker credentials for: {username}")

        full_tag = f"{username}/{image_name.replace('paperforge-', '')}:{tag}"

        # Login to Docker Hub only if password provided (otherwise use existing session)
        if password:
            log("push", "Logging in to Docker Hub...")
            login_result = self._docker_login(username, password)
            logs.append(f"LOGIN:\n{login_result['output']}")

            if not login_result["success"]:
                return PushResult(
                    success=False,
                    error=f"Docker Hub login failed: {login_result['error']}",
                    logs="\n".join(logs)
                )
            log("push", "✓ Logged in to Docker Hub")
        else:
            log("push", "Using existing Docker login session...")

        # Tag the image for Docker Hub
        log("push", f"Tagging image as {full_tag}...")
        tag_result = self._tag_image(image_name, full_tag)
        logs.append(f"TAG:\n{tag_result['output']}")

        if not tag_result["success"]:
            return PushResult(
                success=False,
                error=f"Failed to tag image: {tag_result['error']}",
                logs="\n".join(logs)
            )

        # Push to Docker Hub
        log("push", f"Pushing to Docker Hub (this may take a few minutes)...")
        push_result = self._push_image(full_tag)
        logs.append(f"PUSH:\n{push_result['output']}")

        if not push_result["success"]:
            return PushResult(
                success=False,
                error=f"Failed to push image: {push_result['error']}",
                logs="\n".join(logs)
            )

        image_url = f"https://hub.docker.com/r/{username}/{image_name.replace('paperforge-', '')}"
        log("push", f"✓ Image pushed successfully!")
        log("push", f"Docker Hub URL: {image_url}")
        log("push", f"Pull command: docker pull {full_tag}")

        return PushResult(
            success=True,
            image_url=image_url,
            full_tag=full_tag,
            logs="\n".join(logs)
        )

    def _get_logged_in_user(self) -> Optional[str]:
        """Get currently logged in Docker Hub username from config."""
        import json
        from pathlib import Path
        import os

        # Check Docker config file for stored credentials
        docker_config_paths = [
            Path.home() / ".docker" / "config.json",
            Path(os.environ.get("DOCKER_CONFIG", "")) / "config.json" if os.environ.get("DOCKER_CONFIG") else None
        ]

        for config_path in docker_config_paths:
            if config_path and config_path.exists():
                try:
                    config = json.loads(config_path.read_text())
                    auths = config.get("auths", {})

                    # Check for Docker Hub auth
                    for registry in ["https://index.docker.io/v1/", "docker.io", "registry-1.docker.io"]:
                        if registry in auths:
                            # Auth exists - try to get username from whoami
                            result = subprocess.run(
                                ["docker", "info", "--format", "{{.RegistryConfig.IndexConfigs}}"],
                                capture_output=True,
                                text=True,
                                timeout=10
                            )
                            if result.returncode == 0:
                                # Fallback: ask user to provide username since we can detect login but not username easily
                                return None
                except:
                    pass

        return None

    def _docker_login(self, username: str, password: str) -> dict:
        """Login to Docker Hub."""
        try:
            result = subprocess.run(
                ["docker", "login", "-u", username, "--password-stdin"],
                input=password,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=60
            )
            stdout = result.stdout or ""
            stderr = result.stderr or ""
            return {
                "success": result.returncode == 0,
                "output": stdout + stderr,
                "error": stderr if result.returncode != 0 else None
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": str(e)
            }

    def _tag_image(self, source_image: str, target_tag: str) -> dict:
        """Tag a Docker image for push."""
        try:
            result = subprocess.run(
                ["docker", "tag", source_image, target_tag],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=30
            )
            stdout = result.stdout or ""
            stderr = result.stderr or ""
            return {
                "success": result.returncode == 0,
                "output": stdout + stderr,
                "error": stderr if result.returncode != 0 else None
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": str(e)
            }

    def _push_image(self, image_tag: str) -> dict:
        """Push image to Docker registry."""
        try:
            result = subprocess.run(
                ["docker", "push", image_tag],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=600  # 10 min timeout for large images
            )
            stdout = result.stdout or ""
            stderr = result.stderr or ""
            return {
                "success": result.returncode == 0,
                "output": stdout + stderr,
                "error": stderr if result.returncode != 0 else None
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "output": "",
                "error": "Push timed out after 10 minutes"
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": str(e)
            }

    def build_and_push(
        self,
        project_dir: Path,
        project_name: str,
        dockerhub_username: str,
        dockerhub_password: str = None,
        tag: str = "latest",
        on_progress: Optional[Callable[[str, str], None]] = None
    ) -> PushResult:
        """
        Build image locally and push to Docker Hub in one step.

        Args:
            project_dir: Path to the project
            project_name: Name for the image
            dockerhub_username: Docker Hub username
            dockerhub_password: Docker Hub password/token
            tag: Image tag (default: latest)
            on_progress: Progress callback

        Returns:
            PushResult with status and image URL
        """
        def log(stage: str, msg: str):
            if on_progress:
                on_progress(stage, msg)

        logs = []
        container_name = self._sanitize_name(project_name)
        image_name = f"paperforge-{container_name}"

        # Build the image first
        log("build", f"Building Docker image: {image_name}...")
        build_result = self._build_image(Path(project_dir), image_name)
        logs.append(f"BUILD:\n{build_result['output']}")

        if not build_result["success"]:
            return PushResult(
                success=False,
                error=f"Docker build failed: {build_result['error']}",
                logs="\n".join(logs)
            )

        log("build", "✓ Image built successfully")

        # Now push to Docker Hub
        push_result = self.push_to_dockerhub(
            image_name=image_name,
            username=dockerhub_username,
            password=dockerhub_password,
            tag=tag,
            on_progress=on_progress
        )

        push_result.logs = "\n".join(logs) + "\n" + push_result.logs
        return push_result

    def _sanitize_name(self, name: str) -> str:
        """Convert to valid Docker container name."""
        name = re.sub(r"[^a-z0-9-]", "-", name.lower())
        name = re.sub(r"-+", "-", name)
        return name.strip("-")[:50]

    def _build_image(self, project_dir: Path, image_name: str) -> dict:
        """Build Docker image from project directory."""
        try:
            result = subprocess.run(
                ["docker", "build", "-t", image_name, "."],
                cwd=str(project_dir),
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',  # Handle invalid UTF-8 bytes
                timeout=300  # 5 min timeout for build
            )
            stdout = result.stdout or ""
            stderr = result.stderr or ""
            return {
                "success": result.returncode == 0,
                "output": stdout + stderr,
                "error": stderr if result.returncode != 0 else None
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "output": "",
                "error": "Build timed out after 5 minutes"
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": str(e)
            }

    def _run_container(
        self,
        image_name: str,
        container_name: str,
        port: int
    ) -> dict:
        """Run Docker container."""
        try:
            result = subprocess.run(
                [
                    "docker", "run",
                    "-d",  # Detached mode
                    "--name", container_name,
                    "-p", f"{port}:{port}",
                    "-e", f"PORT={port}",
                    "--restart", "unless-stopped",
                    image_name
                ],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=30
            )

            stdout = result.stdout or ""
            stderr = result.stderr or ""
            container_id = stdout.strip()[:12] if result.returncode == 0 and stdout else None

            return {
                "success": result.returncode == 0,
                "output": stdout + stderr,
                "error": stderr if result.returncode != 0 else None,
                "container_id": container_id
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": str(e)
            }

    def _stop_container(self, container_name: str) -> bool:
        """Stop and remove a container by name."""
        try:
            # Stop container
            subprocess.run(
                ["docker", "stop", container_name],
                capture_output=True,
                timeout=30
            )
            # Remove container
            subprocess.run(
                ["docker", "rm", container_name],
                capture_output=True,
                timeout=10
            )
            return True
        except:
            return False

    def _free_port(self, port: int) -> bool:
        """Stop any container using the specified port."""
        try:
            # Find containers using this port
            result = subprocess.run(
                ["docker", "ps", "-q", "--filter", f"publish={port}"],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=10
            )
            container_ids = (result.stdout or "").strip().split()

            for container_id in container_ids:
                if container_id:
                    subprocess.run(["docker", "stop", container_id], capture_output=True, timeout=30)
                    subprocess.run(["docker", "rm", container_id], capture_output=True, timeout=10)

            return True
        except:
            return False

    def _wait_for_healthy(
        self,
        container_name: str,
        port: int,
        timeout: int = 30
    ) -> bool:
        """Wait for container to respond to health check."""
        import urllib.request
        import urllib.error

        start_time = time.time()
        health_url = f"http://localhost:{port}/health"

        while time.time() - start_time < timeout:
            # First check if container is still running
            try:
                result = subprocess.run(
                    ["docker", "inspect", "-f", "{{.State.Running}}", container_name],
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    timeout=5
                )
                if (result.stdout or "").strip() != "true":
                    return False
            except:
                pass

            # Try health endpoint
            try:
                req = urllib.request.urlopen(health_url, timeout=2)
                if req.status == 200:
                    return True
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ConnectionError, OSError):
                pass
            except Exception:
                # Catch any other connection errors (RemoteDisconnected, etc.)
                pass

            time.sleep(1)

        return False

    def _get_container_logs(self, container_name: str, tail: int = 50) -> str:
        """Get recent logs from container."""
        try:
            result = subprocess.run(
                ["docker", "logs", "--tail", str(tail), container_name],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=10
            )
            return (result.stdout or "") + (result.stderr or "")
        except:
            return ""

    def stop(self, container_name: str) -> bool:
        """Stop a running deployment."""
        return self._stop_container(container_name)

    def get_status(self, container_name: str) -> dict:
        """Get status of a deployed container."""
        try:
            result = subprocess.run(
                [
                    "docker", "inspect",
                    "-f", "{{.State.Status}}|{{.State.Running}}|{{.NetworkSettings.Ports}}",
                    container_name
                ],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=5
            )

            if result.returncode != 0:
                return {"running": False, "status": "not found"}

            parts = (result.stdout or "").strip().split("|")
            return {
                "status": parts[0] if len(parts) > 0 else "unknown",
                "running": parts[1] == "true" if len(parts) > 1 else False,
                "ports": parts[2] if len(parts) > 2 else ""
            }
        except:
            return {"running": False, "status": "error"}

    def list_deployments(self) -> list[dict]:
        """List all PaperForge AI deployments."""
        try:
            result = subprocess.run(
                [
                    "docker", "ps", "-a",
                    "--filter", "name=paperforge-",
                    "--format", "{{.Names}}|{{.Status}}|{{.Ports}}"
                ],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=10
            )

            deployments = []
            for line in (result.stdout or "").strip().split("\n"):
                if line:
                    parts = line.split("|")
                    deployments.append({
                        "name": parts[0] if len(parts) > 0 else "",
                        "status": parts[1] if len(parts) > 1 else "",
                        "ports": parts[2] if len(parts) > 2 else ""
                    })

            return deployments
        except:
            return []


# Convenience functions
def deploy_to_docker(
    project_dir: Path,
    project_name: str,
    port: int = 5001,
    on_progress: Optional[Callable[[str, str], None]] = None
) -> DeploymentResult:
    """
    Deploy a generated project to Docker locally.

    Args:
        project_dir: Path to the generated project
        project_name: Name for the deployment
        port: Port to expose
        on_progress: Progress callback

    Returns:
        DeploymentResult with URL and status
    """
    deployer = DockerDeployer(default_port=port)
    return deployer.deploy(
        project_dir=project_dir,
        project_name=project_name,
        port=port,
        on_progress=on_progress
    )


def push_to_dockerhub(
    image_name: str,
    username: str,
    password: str = None,
    tag: str = "latest",
    on_progress: Optional[Callable[[str, str], None]] = None
) -> PushResult:
    """
    Push an existing local image to Docker Hub.

    Args:
        image_name: Local image name
        username: Docker Hub username
        password: Docker Hub password/token (optional if already logged in)
        tag: Image tag
        on_progress: Progress callback

    Returns:
        PushResult with image URL and status
    """
    deployer = DockerDeployer()
    return deployer.push_to_dockerhub(
        image_name=image_name,
        username=username,
        password=password,
        tag=tag,
        on_progress=on_progress
    )


def build_and_push_to_dockerhub(
    project_dir: Path,
    project_name: str,
    username: str,
    password: str = None,
    tag: str = "latest",
    on_progress: Optional[Callable[[str, str], None]] = None
) -> PushResult:
    """
    Build and push a project to Docker Hub in one step.

    Args:
        project_dir: Path to the project
        project_name: Name for the image
        username: Docker Hub username
        password: Docker Hub password/token
        tag: Image tag
        on_progress: Progress callback

    Returns:
        PushResult with image URL and status
    """
    deployer = DockerDeployer()
    return deployer.build_and_push(
        project_dir=project_dir,
        project_name=project_name,
        dockerhub_username=username,
        dockerhub_password=password,
        tag=tag,
        on_progress=on_progress
    )
