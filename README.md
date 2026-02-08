# pycodei

This project provides an analytics agent running in Python that works with large-scale language models trained by OpenAI. The agent is designed to run Python code in a notebook environment, enabling data analysis, machine learning, inference, and visualization.

## Features

- **Data Analysis and Visualization**: Analyzes datasets and creates graphs suitable for visualization.

- **Trend Extraction and Future Predictions**: Extracts trends from data and provides future predictions.

- **Wide Range of Scientific Topics**: Provides information on a variety of scientific topics, including NLP, machine learning, mathematics, physics, chemistry, and biology.

- **Agent Instructions**: Provides instructions for specifying where data is stored, how to load it, and report creation rules.

- **Memory Rewind**: Provides the ability to go back and redo agent actions.

- **Notebook**: Provides the ability to run saved notebooks to reproduce and modify analyses.

## Installation

### Quick start (PyPI)
```bash
pip install pycodei
```
This installs the published package and exposes the `pycodei` CLI on your PATH.

## Usage
To use the Python Code Interpreter, run the following command (replace the prompt text with your task):
```bash
pycodei
```
Use `pycodei --help` to see optional flags such as `--deployment-name`, which overrides the value in `~/.pycodei/config.json`.

### Command-line Arguments

The following table describes the `pycodei` command-line arguments.

| Argument | Description | Example / Default | Required |
|---|---|---:|:---:|
| `message` (positional) | Initial instruction sent to the interpreter. If omitted, the CLI prompts interactively. | `pycodei "Analyze ./diagnosis.csv"` | No |
| `--version` | Show version and exit. | `pycodei --version` | No |
| `--deployment-name` | Override the `DEPLOYMENT_NAME` environment variable / configuration. If neither is set, the command will error. | `--deployment-name gpt-5-mini` | Conditional (if not set in config) |
| `--load-message` | Path to a message log JSON file to load conversation history (keeps system message). Used for memory rewind. | `--load-message /path/to/log.json` | No |
| `--load-message-without-system` | Path to a message log JSON file to load conversation history but exclude the system message.  Used for memory rewind. | `--load-message-without-system /path/to/log.json` | No |

Note: `Conditional` means required only when the corresponding config or environment variable is not set.

## Configuration
Runtime credentials are read from `~/.pycodei/config.json`. Example:

```json
{
  "DEPLOYMENT_NAME": "gpt-5-mini",
  "PYCODEI_CLIENT": "azure",
  "AZURE_OPENAI_API_KEY": "<your azure openai api key>",
  "AZURE_OPENAI_ENDPOINT": "https://<your endpoint>.openai.azure.com/",
  "OPENAI_API_VERSION": "2024-10-01-preview",
  "OPENAI_API_KEY": "<your openai api key>",
  "Title": "PYCODEI",
  "TitleFont": "smslant",
  "mcpServers": {
    "playwright": {
      "command": "npx",
      "args": [
        "@playwright/mcp@latest"
      ]
    }
  }
}
```

### Configuration Parameters
Set `"PYCODEI_CLIENT"` to `azure` (default) or `openai` to choose which SDK client the interpreter instantiates; you can temporarily override it with the `PYCODEI_CLIENT` environment variable. All other keys map directly to the environment variables expected by the OpenAI/Azure SDKs. Leaving a value blank may cause authentication failures, so be sure to populate the entries relevant to your deployment.

| Key | Description | Example / Default | Required |
|------|-------------|-------------------|----------|
| `DEPLOYMENT_NAME` | The LLM deployment name to use | `gpt-5-mini` | Yes |
| `PYCODEI_CLIENT` | The SDK client to instantiate (`azure` or `openai`) | `azure` or `openai` | Yes |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key (used when `PYCODEI_CLIENT` is `azure`) | `<your azure openai api key>` | Conditional (Azure) |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint URL | `https://<your endpoint>.openai.azure.com/` | Conditional (Azure) |
| `OPENAI_API_VERSION` | OpenAI API version (Azure deployments often require this) | `2024-10-01-preview` | Conditional (Azure) |
| `OPENAI_API_KEY` | OpenAI API key (used when `PYCODEI_CLIENT` is `openai`) | `<your openai api key>` | Conditional (OpenAI) |
| `Title` | Title displayed in the UI | `PYCODEI` | Optional |
| `TitleFont` | [pyfiglet](https://www.figlet.org/examples.html) font style for the title | `smslant` | Optional |
| `mcpServers` | Model Context Protocol server configuration (JSON object) | `{ "playwright": {...} }` | Optional |

Note: `Yes` = required; `Conditional` = required depending on `PYCODEI_CLIENT` selection; `Optional` = not required.

#### Optional: MCP servers
`pycodei` can now connect to multiple [Model Context Protocol](https://modelcontextprotocol.io/) servers using the same schema. Add them under the `"mcpServers"` key in `~/.pycodei/config.json` and set `"disabled": false` for the entries you want to enable. Example:

```json
"mcpServers": {
  "filesystem": {
    "disabled": false,
    "command": "npx",
    "args": [
      "-y",
      "@modelcontextprotocol/server-filesystem",
      "/Users/me/workspace"
    ],
    "env": {
      "FS_ROOT": "/Users/me/workspace"
    }
  },
  "custom_tool": {
    "command": "/usr/local/bin/python",
    "args": [
      "/path/to/server.py"
    ],
    "cwd": "/path/to",
    "transport": "stdio"
  }
}
```

When `pycodei` starts it will spin up each enabled server, discover its MCP tools, and register them as callable functions. Tool names are derived from `server_name__tool_name` and become available to the model alongside the built-in tool. SSE (`transport: "sse"`) and WebSocket (`transport: "websocket"`) endpoints are also supported if you prefer remote servers—specify a `url` instead of a `command`.

## Optional: PYCODEI.md
This is an agent instruction. If you keep a `PYCODEI.md` file, its contents are appended to the system prompt every time `pycodei` starts. Place the file in one of these locations (checked in order): `~/.pycodei/PYCODEI.md`, the directory where your current working directory. Use it for persistent guardrails, safety rules, or project-specific requirements. If you have a database, define the database tables.

## Output
The pycodei outputs files to the logs and notebook directories regardless of interaction.

### logs directory
Saves the memory executed by the agent in the current log directory, which can be used for memory rewind.

### notebook directory
The agent saves the Python code it executed in a notebook in the current notebook directory, which data scientists can later modify and analyze the data.

## Local development workflow
1. Clone the repository:
    ```bash
    git clone https://github.com/KentaroAOKI/python_code_interpreter.git
    cd python_code_interpreter
    ```

2. Install in editable mode (also registers the CLI):
    ```bash
    pip install -e .
    ```

3. Create your user configuration file:
    ```bash
    mkdir -p ~/.pycodei
    cat > ~/.pycodei/config.json <<'EOF'
    {
      "DEPLOYMENT_NAME": "gpt-5-mini",
      "PYCODEI_CLIENT": "azure",
      "AZURE_OPENAI_API_KEY": "<your azure openai api key>",
      "AZURE_OPENAI_ENDPOINT": "https://<your endpoint>.openai.azure.com/",
      "OPENAI_API_VERSION": "2024-10-01-preview",
      "OPENAI_API_KEY": "<your openai api key>"
    }
    EOF
    ```
    Running `pycodei --help` will also create this file with placeholder values if it does not exist.

## Example
You can also start without an inline prompt; the CLI will ask for one interactively:

```bash
# Start CLI and respond to the input prompt
pycodei
```
A notebook file with the execution date and time is created in the results directory. You can view it with Visual Studio Code's Jupyter extension.

- results/result_20250113-082727.ipynb  
Notebook where data analysis steps were recorded.
- results/result_20250113-082727.json  
Log of messages sent to API.

### Example 1 (gpt-3.5-turbo)
"Retrieve NVIDIA stock prices from Yahoo for January to March 2023 and predict prices from April 2023 onwards."  
Result: [sample_01.ipynb](https://github.com/KentaroAOKI/python_code_interpreter/blob/main/sample_results/sample_01.ipynb)

### Example 2 (gpt-3.5-turbo)
"Determine whether the data in /mnt/data/diagnosis.csv is malignant or benign. To make a decision, use the model learned using the load_breast_cancer data available from scikit-learn."  
Result: [sample_02.ipynb](https://github.com/KentaroAOKI/python_code_interpreter/blob/main/sample_results/sample_02.ipynb)

### Example 3 (gpt-3.5-turbo)
"2023年の1月から3月のNVIDIA株価をYahooから取得して、2023年4月以降の株価を予測してください。"  
Result: [sample_03.ipynb](https://github.com/KentaroAOKI/python_code_interpreter/blob/main/sample_results/sample_03.ipynb)

### Example 4 (gpt-3.5-turbo)
"/mnt/data/diagnosis.csv のデータが悪性か良性か判断してください。判断は、scikit-learn から取得できる load_breast_cancer データで学習したモデルを使ってください。日本語で説明してください。"  
Result: [sample_04.ipynb](https://github.com/KentaroAOKI/python_code_interpreter/blob/main/sample_results/sample_04.ipynb)

## License

This project is licensed under the MIT License. See the LICENSE file for details.
