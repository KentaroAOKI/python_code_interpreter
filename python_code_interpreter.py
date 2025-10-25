import argparse
import datetime
import inspect
import json
import os
import re
from pathlib import Path

from openai import AzureOpenAI
from openai import OpenAI
from openai.types.chat import completion_create_params

import openai._utils._transform

import python_code_notebook

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
        self.result_file = ""
        self.messages = []
        base_system_content = (
            f"You are interacting with {self.deployment_name}, a large language model. "
            "The model is based on ReAct technology and uses Python for data analysis and visualization.\n"
            "When a message containing Python code is sent to Python, it is executed in the state-preserving "
            "Jupyter notebook environment. Python returns the results of the execution. "
            f"'{self.persistent_data_dir}' drive can be used to store and persist user files.\n"
            "Python is used to analyze, visualize, and predict the data. If you provide a data set, "
            "we will analyze it and create appropriate graphs for visualization. Additionally, "
            "we can extract trends from the data and provide future projections.\n"
            "We can also provide information on a wide range of scientific topics, "
            "including natural language processing (NLP), machine learning, mathematics, physics, chemistry, "
            "and biology. Let us know what questions you have, what your research needs are, or what problems "
            "you need solved.\n"
            "When a user hands you a file, first understand the type of data you are dealing with, its structure "
            "and characteristics, and tell me its contents. Use clear text and sometimes diagrams.\n"
        )
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
                    # "description": "When there is information I don't know, I run some Python code to get the results.",
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
                            }
                        },
                        "required": ["python_code"],
                    },
                }
            },
        ]
        self.available_functions = {
            "run_python": self.run_python_code_in_notebook,
        } 
    
    def ask_continue(self):
        result = False
        user_input = input("Do you want to continue running this program?(yes/no): ").strip().lower()
        if user_input == "yes":
            result = True
        return result

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

    # Run Python code in the notebook
    def run_python_code_in_notebook(self, code: str, messages):
        if re.match(r'^\s*\{\s*"python_code"\s*:', code):
        # if code.startswith('{"python_code":'):
            code = json.loads(code)["python_code"]

        # Pause to review the program
        print(f"----------\n{code}\n----------")
        if not self.ask_continue():
            quit()

        # Run
        results, self.ipynb_file = python_code_notebook.run_all(
            code,
            messages = messages,
            prepared_notebook=self.ipynb_file,
            result_ipynb_prefix=self.ipynb_prefix,
            remove_result_ipynb=False
            )
        result = results[-1]
        print(result)
        result_strs =  [x['text/plain'] for x in result if x.get('text/plain')]
        result_str = "\n".join(result_strs)
        return result_str

    def write_messages_in_notebook(self, messages):
        result, self.ipynb_file = python_code_notebook.run_all(
            "",
            messages = messages,
            prepared_notebook=self.ipynb_file,
            result_ipynb_prefix=self.ipynb_prefix,
            remove_result_ipynb=False
            )
        return

    def run_conversation(self, message):
        # Create a directory to store persistent data
        if not os.path.exists(self.persistent_data_dir):
            os.makedirs(self.persistent_data_dir, exist_ok=True)
        # Initialize the messages
        if self.system_message is True:
            self.messages.extend(self.messages_system)
        self.messages.append({"role": "user", "content": message})
        # Initialize variables
        max_loops = 200
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
                # tool_choice="auto" if tool_choice_flag else "none", 
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
                    function_response = function_to_call(function_arguments, content_messages)

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
            elif response_reason == 'stop':
                self.write_messages_in_notebook([response_message])
                self.messages.append(
                    {
                        "role": response_role,
                        "content": response_message.content,
                    }
                )
                print(f"Response: {response_message.content}")
                user_input = input("Please write the message you want to send. If you want to finish the conversation, type 'exit': ").strip().lower()
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
        
        # write results
        current_ipynb_file = self.ipynb_file
        self.ipynb_file = os.path.join(os.getcwd(), self.ipynb_result_dir, f'{result_name}.ipynb')
        os.rename(current_ipynb_file, self.ipynb_file)
        self.result_file = os.path.join(os.getcwd(), self.ipynb_result_dir, f'{result_name}.json')
        with open(self.result_file, 'w', encoding='utf-8') as f:
            request_body = openai._utils._transform.maybe_transform({
                "messages": self.messages,
                "model": self.deployment_name,
                "tools": self.tools
            }, completion_create_params.CompletionCreateParamsNonStreaming)
            json.dump(request_body, f, ensure_ascii=False)
        return self.messages

def main(argv=None):
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
    print(assistant_response)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
