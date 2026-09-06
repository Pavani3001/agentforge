# AgentForge

AgentForge builds a specialized AI agent, runs it against an objective,
evaluates the result, and generates an improved specification when the
evaluation fails.

## Requirements

- Python 3.10 or newer
- An API key for an OpenAI-compatible chat-completions provider

The application uses only Python's standard library at runtime.

## Run locally

Set the API key in your PowerShell session. Secrets are read from environment
variables and are never stored by the application.

### OpenAI

```powershell
$env:OPENAI_API_KEY = "your-openai-key"
$env:OPENAI_MODEL = "gpt-4o-mini"
$env:OPENAI_BASE_URL = "https://api.openai.com/v1"
python app.py
```

### OpenRouter

Use an OpenRouter key for slash-qualified model IDs such as
`openai/gpt-oss-20b`:

```powershell
$env:OPENAI_API_KEY = "your-openrouter-key"
$env:OPENAI_MODEL = "openai/gpt-oss-20b"
$env:OPENAI_BASE_URL = "https://openrouter.ai/api/v1"
python app.py
```

When `OPENAI_BASE_URL` is omitted, AgentForge uses
`https://openrouter.ai/api/v1` for model names containing `/`, and
`https://api.openai.com/v1` for other model names. Set the base URL explicitly
when using another OpenAI-compatible provider.

Open http://127.0.0.1:8000 after the server starts. To stop the server, press
`Ctrl+C`.

## API

The web interface calls `POST /api/run` with:

```json
{
	"objective": "Research a topic",
	"criteria": ["Accurate", "Cited"]
}
```

`criteria` may also be sent as a newline-separated string. A successful
response contains the generated `agent_spec`, execution `output`,
`evaluation`, and an `improved_spec` when the evaluation does not pass.

## Troubleshooting

- `OPENAI_API_KEY is not configured`: set the key in the same PowerShell
  session used to start `python app.py`.
- `LLM request failed (403)`: verify that the key belongs to the provider in
  `OPENAI_BASE_URL` and that the selected model is available there.
- Address already in use on port `8000`: stop the existing Python server or
  close the process using that port before starting AgentForge again.

## Tests

`python -m unittest discover -s tests -v`.
