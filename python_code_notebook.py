# pip install nbformat papermill
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
import nbformat
import papermill_enhancement.papermill as pm
import tempfile
import os
import json
import ast
import openai
from pathlib import Path

def run_all(code: str, messages=[], prepared_notebook="", notebook_dir=""):
    if not prepared_notebook:
        prepared_notebook = "notebook.ipynb"
    if not os.path.isabs(prepared_notebook):
        prepared_notebook = os.path.join(notebook_dir, prepared_notebook)
    prepared_notebook = os.path.abspath(prepared_notebook)
    os.makedirs(os.path.dirname(prepared_notebook) or ".", exist_ok=True)
    if os.path.isfile(prepared_notebook):
        nb = nbformat.read(prepared_notebook, as_version=4)
    else:
        nb = new_notebook()
    tmp_in_path = prepared_notebook
    fd, tmp_out_path = tempfile.mkstemp(prefix=Path(prepared_notebook).stem, suffix=".ipynb")
    os.close(fd)
    for message in messages:
        # Add messages as markdown cells
        if isinstance(message, dict):
            nb.cells.append(new_markdown_cell(f"## {message['role']}  \n{message['content']}"))
        elif isinstance(message, openai.types.chat.chat_completion_message.ChatCompletionMessage):
            nb.cells.append(new_markdown_cell(f"## {message.role}  \n{message.content}"))

    if code != "":
        # Add the code as a code cell
        nb.cells.append(new_code_cell(code))
        with open(tmp_in_path, 'w') as f:
            f.write(nbformat.v4.writes(nb))
        result = []
        try:
            # Execute the notebook
            nb = pm.execute_notebook(
                tmp_in_path,
                tmp_out_path,
                kernel_name='Python3',
            )
            # Collect code execution results from the executed notebook
            for cell in nb.cells:
                if cell['cell_type'] == 'code' and 'outputs' in cell:
                    result_cell = []
                    for output in cell['outputs']:
                        if output['output_type'] == 'execute_result' and 'data' in output:
                            ## Collect code execution results, (For example, data.)
                            data = str(output['data'])
                            data_dict = ast.literal_eval(data)
                            data_dict['output_type'] = output['output_type']
                            result_cell.append(data_dict)
                        elif output['output_type'] == 'display_data' and 'data' in output:
                            ## Collect code execution results, (For example, data.)
                            data = str(output['data'])
                            data_dict = ast.literal_eval(data)
                            data_dict['output_type'] = output['output_type']
                            result_cell.append(data_dict)
                        elif output['output_type'] == 'stream' and 'text' in output:
                            ## Collect streams output from code execution (For example, warnings, progress information, etc.)
                            data = {'text/plain': str(output['text']), 'output_type': output['output_type']}
                            result_cell.append(data)
                        elif output['output_type'] == 'error' and 'traceback' in output:
                            ## Collect error details (For example, trace etc.)
                            traces = [output['traceback'][1],output['traceback'][2],output['traceback'][-1]]
                            data = {'text/plain': "\n".join(traces), 'output_type': output['output_type']}
                            result_cell.append(data)
                        else:
                            print(f'output_type={output["output_type"]}')
                        # print(data)
                    result.append(result_cell)
                else:
                    # Pass because cells such as markdown are not included in the result
                    pass

        except Exception as e:
            result.append([{'Exception': str(e)}])
    else:
        with open(tmp_out_path, 'w') as f:
            f.write(nbformat.v4.writes(nb))
        result = []

    # Update content of tmp_in_path to keep file descriptor correct
    with open(tmp_out_path, 'r') as f:
        out_content = f.read()
    with open(tmp_in_path, 'w') as f:
        f.seek(0)
        f.write(out_content)

    return result, tmp_in_path

if __name__ == '__main__':
    import json

    # Set up Python code to run in the notebook
    code = """
import yfinance as yf
import time
from datetime import datetime, timedelta

# Get today's date
end_date = datetime.now()

# Calculate the date one week ago
start_date = end_date - timedelta(days=7)

# Get data for Apple for the past week
data = yf.download('AAPL', start=start_date, end=end_date)
data
"""
    result, result_file = run_all(code, remove_result_ipynb=False)
    print(json.dumps(result[-1]))
    print(result_file)