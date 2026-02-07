"""
PaperForge AI - Web Application
Flask server with real-time progress streaming
Generates code, installs deps, runs server, provides live URL
"""
import os
from dotenv import load_dotenv

# Load .env file before anything else
load_dotenv()

import json
import queue
import shutil
import threading
import uuid
import atexit
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response, send_file
from werkzeug.utils import secure_filename

from src.pdf_parser import extract_text_from_pdf, extract_from_text_file
from src.analyzer import PaperAnalyzer
from src.flask_generator import FlaskGenerator
from src.express_generator import ExpressGenerator
from src.simple_generator import SimpleGenerator, run_simple_pipeline
from src.storage import ProjectStorage
from src.runner import runner
from src.deployer import DockerDeployer, deploy_to_docker
from src.config import get_openai_model, get_gemini_model, get_model_for_provider

app = Flask(__name__)
app.config['SECRET_KEY'] = 'paperforge-ai-secret-key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Ensure folders exist
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)
Path(app.config['OUTPUT_FOLDER']).mkdir(exist_ok=True)

# Store for SSE message queues (per session)
progress_queues: dict[str, queue.Queue] = {}

# Cleanup on exit
atexit.register(runner.stop_all)


def get_api_key(provider: str = "openai") -> str:
    """Get API key for the specified provider."""
    if provider == "gemini":
        return os.environ.get("GEMINI_API_KEY", "")
    return os.environ.get("OPENAI_API_KEY", "")


def get_available_providers() -> list[str]:
    """Get list of providers with configured API keys."""
    providers = []
    if os.environ.get("OPENAI_API_KEY"):
        providers.append("openai")
    if os.environ.get("GEMINI_API_KEY"):
        providers.append("gemini")
    return providers


def get_default_provider() -> str:
    """Get the default provider from environment or first available."""
    default = os.environ.get("DEFAULT_PROVIDER", "").lower()
    available = get_available_providers()

    if default in available:
        return default
    elif available:
        return available[0]
    return "openai"


def send_progress(session_id: str, event: str, data: dict):
    """Send progress update to client."""
    if session_id in progress_queues:
        progress_queues[session_id].put({
            "event": event,
            "data": data
        })


def process_paper(session_id: str, file_path: Path, language: str, framework: str, provider: str = "openai", mode: str = "full"):
    """
    Process a paper, generate code, install deps, run server.
    Sends real-time progress updates via SSE.

    Args:
        mode: 'simple' for single-prompt generation, 'full' for multi-stage pipeline
    """
    storage = ProjectStorage()
    api_key = get_api_key(provider)

    # Get model from .env config
    model = get_model_for_provider(provider)

    try:
        # Step 1: Extract text
        send_progress(session_id, "step", {
            "step": 1,
            "title": "Extracting Text",
            "status": "running",
            "message": f"Reading {file_path.name}..."
        })

        if file_path.suffix.lower() == ".pdf":
            paper = extract_text_from_pdf(file_path)
        else:
            paper = extract_from_text_file(file_path)

        send_progress(session_id, "log", {
            "message": f"✓ Extracted {paper.char_count:,} characters from {paper.page_count} pages",
            "type": "success"
        })
        send_progress(session_id, "log", {
            "message": f"✓ Found sections: {', '.join(paper.sections.keys())}",
            "type": "info"
        })

        # Show any warnings from extraction
        if hasattr(paper, 'warnings') and paper.warnings:
            for warning in paper.warnings:
                send_progress(session_id, "log", {
                    "message": f"⚠ {warning}",
                    "type": "warning"
                })

        send_progress(session_id, "step", {
            "step": 1,
            "title": "Extracting Text",
            "status": "complete"
        })

        output_dir = Path(app.config['OUTPUT_FOLDER'])
        provider_name = "Gemini" if provider == "gemini" else "OpenAI"

        # Branch based on mode: simple vs full
        if mode == "simple":
            # SIMPLE MODE: Single prompt returns all files as JSON
            send_progress(session_id, "step", {
                "step": 2,
                "title": "AI Generation (Simple)",
                "status": "running",
                "message": f"Generating complete MVP with {provider_name}..."
            })
            send_progress(session_id, "log", {
                "message": f"→ Using simple mode - single AI call for complete MVP...",
                "type": "info"
            })

            def simple_progress(stage, msg):
                send_progress(session_id, "log", {"message": msg, "type": "info"})

            simple_generator = SimpleGenerator(api_key, model, provider=provider)
            simple_result = simple_generator.generate(
                paper=paper,
                output_dir=output_dir,
                language=language,
                framework=framework,
                on_progress=simple_progress
            )

            project_dir = simple_result.project_dir

            if simple_result.success:
                send_progress(session_id, "log", {
                    "message": f"✓ Generated {len(simple_result.files)} files",
                    "type": "success"
                })
            else:
                for error in simple_result.errors:
                    send_progress(session_id, "log", {
                        "message": f"⚠ {error}",
                        "type": "warning"
                    })

            send_progress(session_id, "step", {
                "step": 2,
                "title": "AI Generation (Simple)",
                "status": "complete"
            })

            # Skip step 3 for simple mode (combined into step 2)
            send_progress(session_id, "step", {
                "step": 3,
                "title": f"Generating {language} API",
                "status": "complete",
                "message": "Included in simple mode"
            })

            # Create a minimal analysis object for storage
            from dataclasses import dataclass
            @dataclass
            class SimpleAnalysis:
                title: str = simple_result.project_name
                core_algorithm: str = simple_result.project_name
                summary: str = "Generated via simple mode"
                algorithm_steps: list = None
                def __post_init__(self):
                    self.algorithm_steps = self.algorithm_steps or []

            analysis = SimpleAnalysis()

        else:
            # FULL MODE: Multi-stage pipeline (analyze then generate)
            send_progress(session_id, "step", {
                "step": 2,
                "title": "AI Analysis",
                "status": "running",
                "message": f"Analyzing paper with {provider_name}..."
            })
            send_progress(session_id, "log", {
                "message": f"→ Sending to {model} for analysis...",
                "type": "info"
            })

            analyzer = PaperAnalyzer(api_key, model, provider=provider)
            analysis = analyzer.analyze(paper)

            send_progress(session_id, "log", {
                "message": f"✓ Identified algorithm: {analysis.core_algorithm}",
                "type": "success"
            })
            send_progress(session_id, "log", {
                "message": f"✓ Found {len(analysis.algorithm_steps)} implementation steps",
                "type": "success"
            })
            send_progress(session_id, "step", {
                "step": 2,
                "title": "AI Analysis",
                "status": "complete"
            })

            # Step 3: Generate Web API code
            send_progress(session_id, "step", {
                "step": 3,
                "title": f"Generating {language} API",
                "status": "running",
                "message": f"Creating web API project..."
            })
            send_progress(session_id, "log", {
                "message": f"→ Generating {language} web API...",
                "type": "info"
            })

            if language.lower() == "python":
                generator = FlaskGenerator(api_key, model, provider=provider)
                project_dir = generator.generate(analysis, output_dir)
            else:  # nodejs
                generator = ExpressGenerator(api_key, model, provider=provider)
                project_dir = generator.generate(analysis, output_dir)

            send_progress(session_id, "log", {
                "message": f"✓ Created API project structure",
                "type": "success"
            })
            send_progress(session_id, "log", {
                "message": f"✓ Generated algorithm & API routes",
                "type": "success"
            })
            send_progress(session_id, "step", {
                "step": 3,
                "title": f"Generating {language} API",
                "status": "complete"
            })

        # Step 4: Install dependencies
        send_progress(session_id, "step", {
            "step": 4,
            "title": "Installing Dependencies",
            "status": "running",
            "message": "Installing packages..."
        })

        # Step 5: Run the server
        send_progress(session_id, "step", {
            "step": 5,
            "title": "Starting Server",
            "status": "running",
            "message": "Launching your MVP..."
        })

        # Use runner to install and run
        def on_log(msg, msg_type):
            send_progress(session_id, "log", {"message": msg, "type": msg_type})

        # Record in history first
        project_id = storage.add_project(
            paper_path=str(file_path),
            analysis=analysis,
            output_dir=str(project_dir)
        )

        running_project = runner.install_and_run(
            project_id=project_id,
            project_dir=project_dir,
            language=language,
            api_key=api_key,  # Enable auto-fix on errors
            provider=provider,  # Use same provider for error fixing
            on_log=on_log
        )

        send_progress(session_id, "step", {
            "step": 4,
            "title": "Installing Dependencies",
            "status": "complete"
        })
        send_progress(session_id, "step", {
            "step": 5,
            "title": "Starting Server",
            "status": "complete"
        })

        # Step 6: Docker Deployment (automatic)
        docker_url = None
        docker_container = None
        deployer = DockerDeployer()

        # Check if Dockerfile exists before attempting deployment
        dockerfile_exists = (project_dir / "Dockerfile").exists()

        # Generate Dockerfile if missing (fallback)
        if not dockerfile_exists:
            try:
                from src.templates import generate_project_artifacts
                generate_project_artifacts(
                    project_dir=project_dir,
                    project_name=project_dir.name,
                    language=language,
                    algorithm_name=project_dir.name,
                    summary="Generated MVP",
                    port=running_project.port
                )
                dockerfile_exists = (project_dir / "Dockerfile").exists()
                if dockerfile_exists:
                    send_progress(session_id, "log", {
                        "message": "✓ Generated missing Dockerfile",
                        "type": "success"
                    })
            except Exception as e:
                send_progress(session_id, "log", {
                    "message": f"⚠ Could not generate Dockerfile: {e}",
                    "type": "warning"
                })

        if deployer.check_docker() and dockerfile_exists:
            send_progress(session_id, "step", {
                "step": 6,
                "title": "Docker Deployment",
                "status": "running",
                "message": "Building Docker container..."
            })

            def docker_log(stage, msg):
                send_progress(session_id, "log", {"message": f"[Docker] {msg}", "type": "info"})

            docker_port = running_project.port + 1000  # Use different port for Docker
            docker_result = deploy_to_docker(
                project_dir=project_dir,
                project_name=project_dir.name,
                port=docker_port,
                on_progress=docker_log
            )

            if docker_result.success:
                docker_url = docker_result.url
                docker_container = docker_result.container_name
                send_progress(session_id, "log", {
                    "message": f"✓ Docker container running at {docker_url}",
                    "type": "success"
                })
                send_progress(session_id, "step", {
                    "step": 6,
                    "title": "Docker Deployment",
                    "status": "complete"
                })
            else:
                send_progress(session_id, "log", {
                    "message": f"⚠ Docker deployment skipped: {docker_result.error}",
                    "type": "warning"
                })
                send_progress(session_id, "step", {
                    "step": 6,
                    "title": "Docker Deployment",
                    "status": "skipped"
                })
        else:
            send_progress(session_id, "log", {
                "message": "⚠ Docker not available - skipping container deployment",
                "type": "warning"
            })

        # Create zip for download option
        zip_path = shutil.make_archive(
            str(output_dir / f"{project_dir.name}"),
            'zip',
            project_dir
        )

        # Complete!
        complete_data = {
            "success": True,
            "project_id": project_id,
            "project_name": project_dir.name,
            "download_url": f"/download/{project_dir.name}",
            "live_url": running_project.url,
            "port": running_project.port,
            "analysis": {
                "title": analysis.title,
                "algorithm": analysis.core_algorithm,
                "summary": analysis.summary,
                "steps": analysis.algorithm_steps[:5],
                "language": language
            }
        }

        # Add Docker info if available
        if docker_url:
            complete_data["docker_url"] = docker_url
            complete_data["docker_container"] = docker_container

        send_progress(session_id, "complete", complete_data)

    except Exception as e:
        import traceback
        traceback.print_exc()
        send_progress(session_id, "error", {
            "message": str(e)
        })
    finally:
        # Clean up upload
        try:
            file_path.unlink()
        except:
            pass


@app.route('/')
def index():
    """Render main page."""
    providers = get_available_providers()
    default_provider = get_default_provider()
    return render_template('index.html',
                          has_api_key=len(providers) > 0,
                          providers=providers,
                          default_provider=default_provider)


@app.route('/upload', methods=['POST'])
def upload():
    """Handle file upload and start processing."""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Get options
    language = request.form.get('language', 'python')
    framework = request.form.get('framework', 'api')  # Always generate API
    provider = request.form.get('provider', 'openai')
    mode = request.form.get('mode', 'full')  # 'simple' or 'full'

    # Validate mode
    if mode not in ('simple', 'full'):
        mode = 'full'

    # Check API key for selected provider
    if not get_api_key(provider):
        key_name = "GEMINI_API_KEY" if provider == "gemini" else "OPENAI_API_KEY"
        return jsonify({"error": f"{key_name} not set"}), 400

    # Save uploaded file
    filename = secure_filename(file.filename)
    session_id = str(uuid.uuid4())
    file_path = Path(app.config['UPLOAD_FOLDER']) / f"{session_id}_{filename}"
    file.save(file_path)

    # Create progress queue for this session
    progress_queues[session_id] = queue.Queue()

    # Start processing in background thread
    thread = threading.Thread(
        target=process_paper,
        args=(session_id, file_path, language, framework, provider, mode)
    )
    thread.daemon = True
    thread.start()

    return jsonify({
        "session_id": session_id,
        "message": "Processing started"
    })


@app.route('/progress/<session_id>')
def progress(session_id: str):
    """SSE endpoint for progress updates."""
    def generate():
        if session_id not in progress_queues:
            yield f"data: {json.dumps({'event': 'error', 'data': {'message': 'Invalid session'}})}\n\n"
            return

        q = progress_queues[session_id]
        while True:
            try:
                msg = q.get(timeout=30)
                yield f"data: {json.dumps(msg)}\n\n"

                # Clean up on complete or error
                if msg.get('event') in ('complete', 'error'):
                    del progress_queues[session_id]
                    break
            except queue.Empty:
                # Send keepalive
                yield f"data: {json.dumps({'event': 'ping'})}\n\n"

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive'
        }
    )


@app.route('/download/<project_name>')
def download(project_name: str):
    """Download generated project as zip."""
    zip_path = Path(app.config['OUTPUT_FOLDER']) / f"{project_name}.zip"
    if not zip_path.exists():
        return jsonify({"error": "Project not found"}), 404

    return send_file(
        zip_path,
        mimetype='application/zip',
        as_attachment=True,
        download_name=f"{project_name}.zip"
    )


@app.route('/running')
def running_projects():
    """Get list of running projects."""
    return jsonify(runner.get_running())


@app.route('/stop/<project_id>', methods=['POST'])
def stop_project(project_id: str):
    """Stop a running project."""
    if runner.stop_project(project_id):
        return jsonify({"success": True})
    return jsonify({"error": "Project not found"}), 404


@app.route('/history')
def history():
    """Get generation history."""
    storage = ProjectStorage()
    entries = storage.get_history(limit=20)
    return jsonify(entries)


@app.route('/docker/deploy/<project_name>', methods=['POST'])
def docker_deploy(project_name: str):
    """Deploy a project to local Docker."""
    project_dir = Path(app.config['OUTPUT_FOLDER']) / project_name
    if not project_dir.exists():
        return jsonify({"error": "Project not found"}), 404

    port = request.json.get('port', 5001) if request.json else 5001

    def on_progress(stage, msg):
        print(f"[Docker {stage}] {msg}")

    result = deploy_to_docker(
        project_dir=project_dir,
        project_name=project_name,
        port=port,
        on_progress=on_progress
    )

    if result.success:
        return jsonify({
            "success": True,
            "url": result.url,
            "container_id": result.container_id,
            "container_name": result.container_name,
            "image_name": result.image_name
        })
    else:
        return jsonify({
            "success": False,
            "error": result.error
        }), 500


@app.route('/docker/push/<project_name>', methods=['POST'])
def docker_push(project_name: str):
    """Push a project image to Docker Hub."""
    project_dir = Path(app.config['OUTPUT_FOLDER']) / project_name
    if not project_dir.exists():
        return jsonify({"error": "Project not found"}), 404

    data = request.json or {}
    username = data.get('username')
    password = data.get('password')
    tag = data.get('tag', 'latest')

    if not username:
        return jsonify({"error": "Docker Hub username required"}), 400

    deployer = DockerDeployer()

    def on_progress(stage, msg):
        print(f"[Docker {stage}] {msg}")

    # Build and push
    result = deployer.build_and_push(
        project_dir=project_dir,
        project_name=project_name,
        dockerhub_username=username,
        dockerhub_password=password,
        tag=tag,
        on_progress=on_progress
    )

    if result.success:
        return jsonify({
            "success": True,
            "image_url": result.image_url,
            "full_tag": result.full_tag,
            "pull_command": f"docker pull {result.full_tag}"
        })
    else:
        return jsonify({
            "success": False,
            "error": result.error
        }), 500


@app.route('/docker/status')
def docker_status():
    """Check Docker availability and list deployments."""
    deployer = DockerDeployer()
    docker_available = deployer.check_docker()
    deployments = deployer.list_deployments() if docker_available else []

    return jsonify({
        "docker_available": docker_available,
        "deployments": deployments
    })


if __name__ == '__main__':
    print("\n" + "="*50)
    print("  PaperForge AI - Research Paper to MVP Generator")
    print("="*50)

    providers = get_available_providers()
    default = get_default_provider()
    if not providers:
        print("\n⚠️  WARNING: No API keys configured!")
        print("   Set at least one in your .env file:")
        print("   OPENAI_API_KEY=your-openai-key")
        print("   GEMINI_API_KEY=your-gemini-key")
    else:
        print(f"\n✓ Default provider: {default}")
        print("✓ Available providers:")
        if "openai" in providers:
            print("  - OpenAI (GPT-4o)")
        if "gemini" in providers:
            print("  - Gemini (gemini-2.0-flash)")

    port = int(os.environ.get("PORT", 5000))
    print(f"\n🚀 Starting server at http://localhost:{port}")
    print("="*50 + "\n")

    app.run(debug=False, host='0.0.0.0', port=port)
