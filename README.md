# PaperForge AI

> Transform research papers into runnable MVP code using AI

A web-based tool that analyzes algorithm/systems research papers and generates complete, runnable starter projects in **Python** or **Node.js**.

Now featuring a 3-stage approach (Planning → Analysis → Coding) for higher quality code generation.

## Features

- **Web Interface** - Simple drag & drop upload with real-time progress
- **Multi-Language** - Generate Python or Node.js code
- **Terminal Display** - Watch each processing step in real-time
- **AI-Powered** - Supports OpenAI GPT-4o and Google Gemini for analysis and code generation
- **Download Ready** - Get a complete, runnable project as a ZIP file
- **JSON Storage** - No database needed, everything uses JSON files
- **Docker Deployment** - Auto-build and deploy to Docker with one command

## New: PaperForge AI Style Pipeline

The enhanced pipeline follows PaperForge AI's proven approach:

```
┌─────────────────────────────────────────────────────────────┐
│           PaperForge AI-Style MVP Pipeline                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   1. PLANNING STAGE                                         │
│      - Parse paper content                                  │
│      - Design API endpoints & data models                   │
│      - Create file structure & task breakdown               │
│                                                              │
│   2. ANALYSIS STAGE                                         │
│      - Detailed logic for each file                         │
│      - API endpoint specifications                          │
│      - Class/function signatures                            │
│                                                              │
│   3. CODING STAGE                                           │
│      - Generate files one-by-one                            │
│      - Each file has context from previous files            │
│      - More coherent, integrated code                       │
│                                                              │
│   4. DEBUGGING (Auto-Fix)                                   │
│      - SEARCH/REPLACE format for precise fixes              │
│      - Automatic error detection and correction             │
│                                                              │
│   5. DEPLOYMENT (Docker)                                    │
│      - Build Docker image automatically                     │
│      - Run container and return live URL                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
cd paperforge-ai
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your API key(s)
```

Your `.env` file should contain:
```env
# Choose default provider: "openai" or "gemini"
DEFAULT_PROVIDER=openai

# Set at least one API key
OPENAI_API_KEY=your-openai-key-here
GEMINI_API_KEY=your-gemini-key-here
```

### 3. Run the Server

```bash
python app.py
```

### 4. Open Browser

Navigate to: **http://localhost:5000**

## Usage

1. **Upload** - Drag & drop or click to upload a research paper (PDF or TXT)
2. **Select AI Provider** - Choose OpenAI or Gemini
3. **Select Language** - Choose Python or Node.js
4. **Generate** - Click "Generate & Run MVP" and watch the progress
5. **Access** - Your MVP is live with a URL, or download the code!

## Programmatic Usage

Use the enhanced pipeline directly in Python:

```python
from src import (
    extract_text_from_pdf,
    run_full_pipeline
)

# Extract paper content
paper = extract_text_from_pdf("research_paper.pdf")

# Run the full PaperForge AI-style pipeline
project = run_full_pipeline(
    paper=paper,
    output_dir="./output",
    api_key="your-api-key",
    language="python",
    framework="flask",
    provider="openai",  # or "gemini"
    on_progress=lambda stage, msg: print(f"[{stage}] {msg}")
)

print(f"Project generated at: {project.project_dir}")
print(f"Files generated: {len(project.files)}")
```

## Docker Deployment

Deploy generated projects to Docker automatically:

```python
from src import (
    extract_text_from_pdf,
    run_full_pipeline,
    deploy_to_docker
)

# Generate the project
paper = extract_text_from_pdf("research_paper.pdf")
project = run_full_pipeline(
    paper=paper,
    output_dir="./output",
    api_key="your-api-key",
    language="python",
    framework="flask"
)

# Deploy to Docker
result = deploy_to_docker(
    project_dir=project.project_dir,
    project_name="my-mvp",
    port=5001,
    on_progress=lambda stage, msg: print(f"[{stage}] {msg}")
)

if result.success:
    print(f"Deployed! URL: {result.url}")
    print(f"Container: {result.container_name}")
else:
    print(f"Failed: {result.error}")
```

**Prerequisites**: Docker must be installed and running.

**Manage deployments**:
```python
from src import DockerDeployer

deployer = DockerDeployer()

# List all PaperForge AI deployments
deployments = deployer.list_deployments()

# Stop a deployment
deployer.stop("paperforge-my-mvp")

# Check status
status = deployer.get_status("paperforge-my-mvp")
```

## Generated Project Structure

### Python Project (Flask API)
```
project_name/
├── app.py              # Flask API entry point
├── algorithm.py        # Core algorithm implementation
├── json_storage.py     # JSON-based CRUD storage
├── config.yaml         # Configuration from paper
├── requirements.txt    # Python dependencies
├── Dockerfile          # Docker containerization
├── docker-compose.yml  # Docker orchestration
├── fly.toml            # Fly.io deployment config
├── railway.json        # Railway deployment config
├── render.yaml         # Render deployment config
├── Procfile            # Heroku/Railway procfile
├── README.md           # Documentation
├── run.sh / run.bat    # Run scripts
├── data/               # JSON data storage
└── results/            # Algorithm results
```

### Node.js Project (Express API)
```
project_name/
├── app.js              # Express API entry point
├── algorithm.js        # Core algorithm implementation
├── jsonStorage.js      # JSON-based CRUD storage
├── package.json        # Node.js dependencies
├── Dockerfile          # Docker containerization
├── docker-compose.yml  # Docker orchestration
├── fly.toml            # Fly.io deployment config
├── railway.json        # Railway deployment config
├── README.md           # Documentation
├── data/               # JSON data storage
└── results/            # Algorithm results
```

## Project Architecture

```
paperforge-ai/
├── app.py                  # Flask web server (main entry point)
├── requirements.txt        # Python dependencies
├── .env.example            # Environment template
├── README.md               # This file
├── PAPER2MVP_PLAN.md       # Enhancement roadmap
├── templates/
│   └── index.html          # Web interface
├── src/
│   ├── __init__.py         # Module exports
│   ├── pdf_parser.py       # PDF/TXT text extraction
│   ├── analyzer.py         # Original single-stage analysis
│   ├── mvp_planning.py     # NEW: Multi-stage planning
│   ├── mvp_analyzing.py    # NEW: Detailed file analysis
│   ├── mvp_coding.py       # NEW: Staged code generation
│   ├── flask_generator.py  # Python/Flask generator
│   ├── express_generator.py # Node.js/Express generator
│   ├── error_fixer.py      # ENHANCED: SEARCH/REPLACE debugging
│   ├── runner.py           # Run generated projects
│   ├── storage.py          # JSON project storage
│   ├── templates.py        # Dockerfile, deployment configs
│   ├── prompts.py          # All LLM prompts
│   ├── utils.py            # NEW: Utilities from PaperForge AI
│   └── deployer.py         # NEW: Docker deployment
├── samples/                # Test papers
├── uploads/                # Temporary uploads
├── output/                 # Generated projects
└── storage/                # Project history
```

## Key Enhancements from PaperForge AI

| Feature | Original | Enhanced |
|---------|----------|----------|
| Pipeline | Single-stage | 3-stage (Plan → Analyze → Code) |
| File Generation | All at once | Sequential with context |
| Error Fixing | Replace entire file | SEARCH/REPLACE (minimal changes) |
| JSON Parsing | Basic | Robust multi-fallback parsing |
| Deployment | ZIP only | Auto Docker deploy with live URL |

## Supported Papers

Works best with papers that have:
- Clear algorithm descriptions
- Pseudocode sections
- Well-defined inputs/outputs
- Step-by-step methodology

Good for:
- Sorting/searching algorithms
- Graph algorithms
- Optimization methods
- Data structure implementations
- System design papers
- Web service concepts

## Limitations

- Quality depends on how clearly the paper describes the algorithm
- Complex papers with heavy math notation may need manual refinement
- Generated code is a starting point - review before production use
- Large papers are truncated to fit AI context limits

## API Cost Estimation

Typical costs per paper (using GPT-4o):
- **Planning Stage**: ~$0.02-0.05
- **Analysis Stage**: ~$0.03-0.08
- **Coding Stage**: ~$0.05-0.15
- **Total**: ~$0.10-0.30 per paper

## API Keys

Get your API keys from:
- **OpenAI**: https://platform.openai.com/api-keys (uses GPT-4o)
- **Gemini**: https://aistudio.google.com/apikey (uses gemini-2.0-flash)

Set `DEFAULT_PROVIDER` in your `.env` file to choose which provider to use by default.

## Tech Stack

- **Backend**: Flask (Python)
- **Frontend**: Vanilla HTML/CSS/JS (no build needed)
- **AI**: OpenAI GPT-4o / Google Gemini 2.0 Flash
- **PDF Parsing**: PyMuPDF
- **Real-time Updates**: Server-Sent Events (SSE)
- **Deployment**: Docker, Railway, Fly.io, Render

## License

MIT License
