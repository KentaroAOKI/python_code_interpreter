import argparse
import datetime
import inspect
import json
import os
import re
from pathlib import Path
import textwrap
from colorama import init, Fore, Style

from openai import AzureOpenAI
from openai import OpenAI
from openai.types.chat import completion_create_params

import openai._utils._transform

import python_code_notebook
from mcp_client_manager import MCPClientManager

DEFAULT_CONFIG_DIR = Path.home() / ".pycodei"
CONFIG_DIR = Path(os.environ.get("PYCODEI_CONFIG_DIR", DEFAULT_CONFIG_DIR))
CONFIG_PATH = CONFIG_DIR / "config.json"
GUIDE_FILENAME = "PYCODEI.md"
DEFAULT_CONFIG = {
    "DEPLOYMENT_NAME": "gpt-4o-mini",
    "PYCODEI_CLIENT": "azure",
    "AZURE_OPENAI_API_KEY": "",
    "AZURE_OPENAI_ENDPOINT": "https://<your-endpoint>.openai.azure.com/",
    "OPENAI_API_VERSION": "2024-10-01-preview",
    "OPENAI_API_KEY": "",
    "CONVERSATION_LOOP_MAX_CYCLES": 100,
    "mcpServers": {},
}


def load_user_config():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if not CONFIG_PATH.exists():
        CONFIG_PATH.write_text(json.dumps(DEFAULT_CONFIG, indent=2), encoding="utf-8")
        raise RuntimeError(
            f"Created a config template at {CONFIG_PATH}. "
            "Update it with your deployment name and API credentials."
        )

    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as f:
            config_data = json.load(f)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in {CONFIG_PATH}: {exc}") from exc

    if not isinstance(config_data, dict):
        raise RuntimeError(f"{CONFIG_PATH} must contain a JSON object of key/value pairs.")

    if not config_data.get("DEPLOYMENT_NAME"):
        raise RuntimeError(f"'DEPLOYMENT_NAME' is missing in {CONFIG_PATH}.")

    return config_data


def apply_config_to_env(config_data):
    for key, value in config_data.items():
        if value is None:
            continue
        if isinstance(value, (dict, list)):
            continue
        os.environ[key] = str(value)


def initialize_configuration():
    try:
        config_data = load_user_config()
    except RuntimeError as exc:
        raise SystemExit(exc) from exc
    apply_config_to_env(config_data)
    return config_data


def load_pycodei_guide():
    search_paths = [
        Path.cwd() / GUIDE_FILENAME,
        CONFIG_DIR / GUIDE_FILENAME,
        Path(__file__).resolve().parent / GUIDE_FILENAME,
    ]
    for path in search_paths:
        if not path.exists():
            continue
        try:
            content = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if content:
            return content
    return ""

CONFIG = initialize_configuration()
deployment_name = os.getenv("DEPLOYMENT_NAME")
MCP_MANAGER = MCPClientManager(CONFIG.get("mcpServers"), base_dir=CONFIG_DIR)

def resolve_client_provider():
    provider = (
        os.getenv("PYCODEI_CLIENT")
        or CONFIG.get("PYCODEI_CLIENT")
        or DEFAULT_CONFIG["PYCODEI_CLIENT"]
    )
    return provider.strip().lower()


def create_llm_client():
    provider = resolve_client_provider()
    if provider == "azure":
        return AzureOpenAI()
    if provider == "openai":
        return OpenAI()
    raise SystemExit(
        f"Unsupported PYCODEI_CLIENT '{provider}'. Supported values are 'azure' or 'openai'."
    )

class PythonCodeInterpreter():
    def __init__(self, deployment_name: str):

        self.client_provider = resolve_client_provider()
        self.client = create_llm_client()

        self.system_message = True
        self.deployment_name = deployment_name
        self.persistent_data_dir = os.path.join(os.getcwd(), "ai_workspace")
        self.current_messages_index = 1
        self.ipynb_result_dir = "results"
        self.ipynb_prefix = os.path.join(os.getcwd(), self.ipynb_result_dir, "running_")
        self.ipynb_file = ""
        self.ipynb_dir = os.path.join(os.getcwd(),"notebooks", datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
        self.result_file = ""
        self.messages = []
        self.tool_auto_permissions = {}
        self.tool_function_auto_permissions = set()
        self.tool_descriptions = {}
        base_system_content = textwrap.dedent(f"""
            You are **{self.deployment_name}**, a large language model using **ReAct** reasoning with Python integration for computation, data analysis, and visualization.

            #### **Core Behavior**
            - Combine reasoning and actions (Python execution) to solve tasks.
            - Execute Python in a **stateful Jupyter environment** for analysis and visualization.
            - Use **'{self.persistent_data_dir}'** for persistent file storage.

            #### **Capabilities**
            - Analyze datasets: detect structure, summarize, extract insights.
            - Visualize data: generate clear charts and diagrams.
            - Predict trends and provide projections.
            - Provide accurate information on NLP, ML, math, physics, chemistry, biology.

            #### **File Handling**
            - When a file is provided:
            1. Identify type, structure, and key characteristics.
            2. Summarize contents clearly; use diagrams if helpful.

            #### **Jupyter notebook file as ipynb_file Handling**
            Create a notebook file (.ipynb) in the `./notebooks` directory.
            - For a new purpose:
            1. Name the file appropriately using alphabetic characters.
            2. Use sequential numbering to make the file name different from existing filenames.
            3. Create a new notebook file.
            - For a continuous purpose:
            1. Specify the existing filename.
            2. Add a cell to the end of the existing notebook and run it.

            #### **Safety & Constraints**
            - Always verify reasoning before answering.
            - Ask clarifying questions if user intent is unclear.

            #### **Interaction Guidelines**
            - Explain reasoning steps clearly.
            - Present results in structured formats (tables, bullet points).
            - Use visual aids for complex data when possible.
            """)
        pycodei_guide = load_pycodei_guide()
        if pycodei_guide:
            base_system_content = (
                f"{base_system_content}\nAdditional instructions from {GUIDE_FILENAME}:\n{pycodei_guide}\n"
            )
        self.messages_system = [{
            "role": "system",
            "content": base_system_content
        }]
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "run_python",
                    "description": "If some information is unknown, run Python code to get the data from outside, do calculations, etc., to get the results.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "python_code": {
                                "type": "string",
                                "description": (
                                    "The python program code. The python is used to execute the python code created "
                                    "by you for all information. python code must not perform any directory or file "
                                    "operations outside of the current directory."
                                ),
                            },
                            "ipynb_file": {
                                "type": "string",
                                "description": (
                                    "The path to the Jupyter notebook file to use as the execution environment. "
                                    "If not exist, a new temporary notebook will be created for execution."
                                ),
                            },
                        },
                        "required": ["python_code", "ipynb_file"],
                    },
                }
            },
        ]
        self._register_tool_descriptions(self.tools)
        self.available_functions = {
            "run_python": self.run_python_code_in_notebook,
        } 

        mcp_tools, mcp_function_map = MCP_MANAGER.get_openai_tools()
        if mcp_tools:
            self.tools.extend(mcp_tools)
            self.available_functions.update(mcp_function_map)
            self._register_tool_descriptions(mcp_tools)
    
    def _register_tool_descriptions(self, tools):
        for tool in tools:
            if tool.get("type") != "function":
                continue
            function_info = tool.get("function") or {}
            name = function_info.get("name")
            description = function_info.get("description")
            if name:
                self.tool_descriptions[name] = description or ""

    def _show_tool_request_details(self, function_name, function_arguments):
        description = self.tool_descriptions.get(function_name, "")
        formatted_args = self._format_tool_arguments(function_arguments)
        print("------")
        print(Fore.GREEN + "A tool execution was requested:")
        print(Fore.GREEN + (f"Function : {function_name}"))
        if description:
            print(Fore.GREEN + (f"Description : {description}"))
        print(Fore.GREEN + "Arguments:")
        print(Fore.GREEN + formatted_args.replace("\\n", "\n"))

    def _format_tool_arguments(self, raw_args):
        if raw_args is None:
            return "(no arguments)"
        if isinstance(raw_args, str):
            stripped = raw_args.strip()
            if not stripped:
                return "(empty string)"
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                return raw_args
            return json.dumps(parsed, indent=2, ensure_ascii=False)
        return str(raw_args)

    def _tool_approval_key(self, function_name, raw_args):
        normalized_args = ""
        if raw_args is not None:
            if isinstance(raw_args, str):
                stripped = raw_args.strip()
                try:
                    parsed = json.loads(stripped)
                except json.JSONDecodeError:
                    normalized_args = stripped
                else:
                    normalized_args = json.dumps(parsed, sort_keys=True, separators=(",", ":"))
            else:
                normalized_args = str(raw_args)
        return f"{function_name}:{normalized_args}"

    def _prompt_tool_execution(self, function_name, function_arguments):
        self._show_tool_request_details(function_name, function_arguments)
        print(
            "Options: [y] allow once / [a] always allow this function (with args) / "
            "[f] always allow this function / [n] deny"
        )
        while True:
            decision = input("Enter your choice (y/a/f/n): ").strip().lower()
            if decision in ("y", "yes"):
                return "allow"
            if decision in ("a", "always_function+args"):
                return "always_function+args"
            if decision in ("f", "always_function"):
                return "always_function"
            if decision in ("n", "no"):
                return "deny"
            print("Invalid choice. Please respond with 'y', 'a', 'f', or 'n'.")

    # helper method used to check if the correct arguments are provided to a function
    def check_args(self, function, args):
        sig = inspect.signature(function)
        params = sig.parameters

        # Check if there are extra arguments
        for name in args:
            if name not in params:
                return False
        # Check if the required arguments are provided 
        for name, param in params.items():
            if param.default is param.empty and name not in args:
                return False
        return True

    def run_python_code_in_notebook(self, code: str, messages):
        """Run the provided Python code in a Jupyter notebook environment and return the result as a string."""
        # Extract code from JSON if necessary
        if re.match(r'^\s*\{\s*"python_code"\s*:', code):
            code_json = json.loads(code)
            code = code_json["python_code"]
            self.ipynb_file = code_json.get("ipynb_file", "notebook.ipynb")
        else:
            self.ipynb_file = "notebook.ipynb"

        # Run the code in the notebook
        results, self.ipynb_file = python_code_notebook.run_all(
            code,
            messages = messages,
            prepared_notebook=self.ipynb_file,
            notebook_dir=self.ipynb_dir,
            )
        result = results[-1]
        result_strs =  [x['text/plain'] for x in result if x.get('text/plain')]
        result_str = "\n".join(result_strs)
        return result_str

    def write_messages_in_notebook(self, messages):
        result, self.ipynb_file = python_code_notebook.run_all(
            "",
            messages = messages,
            prepared_notebook=self.ipynb_file,
            notebook_dir=self.ipynb_dir,
            )
        return

    def run_conversation(self, message):
        """Run the conversation loop with the provided initial message."""
        # Create a directory to store persistent data
        if not os.path.exists(self.persistent_data_dir):
            os.makedirs(self.persistent_data_dir, exist_ok=True)
        # Initialize the messages
        if self.system_message is True:
            self.messages.extend(self.messages_system)
        self.messages.append({"role": "user", "content": message})
        # Initialize variables
        max_loops = int(os.getenv("CONVERSATION_LOOP_MAX_CYCLES", 100))
        tool_choice_flag = False
        finish_flag = False
        usage_total_tokens = 0
        result_name = f'result_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
        # Conversations start
        for i in range(max_loops):
            print(f"Loop {i+1}/{max_loops}, total tokens used: {usage_total_tokens}")

            completion = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=self.messages,
                tools=self.tools,
            )

            usage_total_tokens += completion.usage.total_tokens
            response_message = completion.choices[0].message
            response_reason = completion.choices[0].finish_reason
            response_role = response_message.role

            if response_reason == 'tool_calls':
                self.messages.append(response_message)
                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    function_arguments = tool_call.function.arguments

                    if function_name not in self.available_functions:
                        return f"Function {function_name} does not exist."

                    function_to_call = self.available_functions[function_name]
                    if response_message.content is None:
                        content_messages = self.messages[self.current_messages_index:]
                    else:
                        content_messages = [response_message]
                    
                    # Handle tool execution approval
                    approval_key = self._tool_approval_key(function_name, function_arguments)
                    should_execute = True
                    auto_permission_type = None
                    if function_name in self.tool_function_auto_permissions:
                        auto_permission_type = "function"
                    elif approval_key in self.tool_auto_permissions:
                        auto_permission_type = "function+args"

                    if auto_permission_type:
                        self._show_tool_request_details(function_name, function_arguments)
                        print(f"Auto-approved tool call '{function_name}' ({auto_permission_type}-level).")
                    else:
                        decision = self._prompt_tool_execution(function_name, function_arguments)
                        if decision == "deny":
                            should_execute = False
                        elif decision == "always_function+args":
                            self.tool_auto_permissions[approval_key] = True
                        elif decision == "always_function":
                            self.tool_function_auto_permissions.add(function_name)
                    if should_execute:
                        function_response = function_to_call(function_arguments, content_messages)
                    else:
                        function_response = "Tool execution was skipped because you denied the request."

                    # special treatment. For some reason, an error occurs when inserting a figure strings
                    # if function_response.startswith('<Figure size'):
                    #     function_response = "Omitted due to the large size of the image."

                    self.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": function_name,
                            "content": function_response,
                        }
                    )
                    print(f"------\n{Fore.YELLOW}Function: '{function_name}'\nResponse:\n\t{function_response.replace('\\n', '\n\t')}")
            elif response_reason == 'stop':
                self.write_messages_in_notebook([response_message])
                self.messages.append(
                    {
                        "role": response_role,
                        "content": response_message.content,
                    }
                )
                print(f"------\nResponse: {response_message.content}")
                user_input = input("Chat here. 'exit' to leave: ").strip().lower()
                if user_input == 'exit':
                    finish_flag = True
                else:
                    self.write_messages_in_notebook([{"role": "user", "content": user_input}])
                    self.messages.append({"role": "user", "content": user_input})
            elif response_reason == 'length':
                print("Response length exceeded the limit. Please try again with a shorter message.")
            else:
                print(f"Unexpected response reason: {response_reason}")
            self.current_messages_index = len(self.messages)
            if finish_flag:
                break
        
        self.result_file = os.path.join(os.getcwd(), self.ipynb_result_dir, f'{result_name}.json')
        with open(self.result_file, 'w', encoding='utf-8') as f:
            request_body = openai._utils._transform.maybe_transform({
                "messages": self.messages,
                "model": self.deployment_name,
                "tools": self.tools
            }, completion_create_params.CompletionCreateParamsNonStreaming)
            json.dump(request_body, f, ensure_ascii=False)
        print(f"Conversation finished. Request/Response messages: {self.result_file}")
        return self.messages

def main(argv=None):
    init(autoreset=True)
    parser = argparse.ArgumentParser(
        description="Run the Python Code Interpreter conversation loop."
    )
    parser.add_argument(
        "message",
        nargs="?",
        help="Initial instruction sent to the interpreter. If omitted, you will be prompted."
    )
    parser.add_argument(
        "--deployment-name",
        dest="deployment_name",
        default=None,
        help="Override the DEPLOYMENT_NAME environment variable."
    )
    args = parser.parse_args(argv)

    resolved_deployment = args.deployment_name or os.getenv("DEPLOYMENT_NAME")
    if not resolved_deployment:
        parser.error("DEPLOYMENT_NAME is not set. Export it or pass --deployment-name.")

    message = args.message
    if not message:
        try:
            message = input("Enter the initial message for the interpreter: ").strip()
        except KeyboardInterrupt:
            print("\nAborted.")
            return 1

    if not message:
        print("No message provided. Exiting.")
        return 1

    pci = PythonCodeInterpreter(resolved_deployment)
    pci.system_message = True
    assistant_response = pci.run_conversation(message)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
