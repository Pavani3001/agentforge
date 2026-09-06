# AgentForge

Autonomous AI Agent Builder and Improver.

AgentForge creates specialized AI agents, evaluates their performance, identifies failures, and iteratively improves them.

## Run locally

AgentForge uses an OpenAI-compatible chat-completions API. Configure the API key
in the environment; secrets are never stored in the application:

```powershell
$env:OPENAI_API_KEY = "your-key"
$env:OPENAI_MODEL = "gpt-4o-mini" # optional; slash-qualified models default to OpenRouter
$env:OPENAI_BASE_URL = "https://api.openai.com/v1" # optional
python app.py
```

For `openai/gpt-oss-20b` and other slash-qualified model IDs, set
`OPENAI_API_KEY` to the matching provider key. AgentForge defaults those model
IDs to `https://openrouter.ai/api/v1`; set `OPENAI_BASE_URL` explicitly when
using another OpenAI-compatible provider.

Open http://127.0.0.1:8000. The API endpoint is `POST /api/run` with:

```json
{ "objective": "Research a topic", "criteria": ["Accurate", "Cited"] }
```

The pipeline generates a specification, executes it, evaluates the output, and
generates an improved specification when evaluation fails. Run tests with
`python -m unittest discover -s tests -v`.
