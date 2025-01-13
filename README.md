# Python Code Interpreter

This project provides a Python code interpreter that interacts with large language models trained by OpenAI. This interpreter is designed to run Python code in a notebook environment, enabling data analysis, visualization, and prediction. The Python code runs interactively, but in the notebook from the beginning each time.

## Features

- **Data Analysis and Visualization**: Analyze datasets and create appropriate graphs for visualization.
- **Trend Extraction and Future Projections**: Extract trends from data and provide future projections.
- **Wide Range of Scientific Topics**: Provide information on various scientific topics, including NLP, machine learning, mathematics, physics, chemistry, and biology.
- **Secure Handling of User Files**: Store and persist user files in the `/mnt/data` drive.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/KentaroAOKI/python_code_interpreter.git
    cd python_code_interpreter
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up environment variables:
    ```bash
    cp .env.example .env
    # Edit the .env file to include your deployment name and other necessary configurations
    ```
4. Set up data directory:
    ```bash
    sudo ln -s `pwd`/sample_data /mnt/data
    ```

## Usage
To use the Python Code Interpreter, run the following command:
```bash
python python_code_interpreter.py
```
If you are using OpenAI, please modify the following part of python_code_interpreter.py.

```python
        self.client = OpenAI()
        # self.client = AzureOpenAI()
```

### Example
Please change the bottom part of python_code_interpreter.py and enter your analysis content.

```python
message = "Retrieve NVIDIA stock prices from Yahoo for January to March 2023 and predict prices from April 2023 onwards."
pci = PythonCodeInterpreter(deployment_name)
assistant_response = pci.run_conversation(message)
print(json.dumps(assistant_response))
```
A notebook file with the execution date and time is created in the results directory. You can view it with Visual Studio Code's Jupyter extension.

- results/result_20250113-082727.ipynb  
Notebook where data analysis steps were recorded.
- results/result_20250113-082727.json  
Log of messages sent to API.

## Environment Variables
Set environment variables in .env file.

```sh:.env
DEPLOYMENT_NAME='gpt-4o-mini'

# Azure OpenAI
AZURE_OPENAI_API_KEY='<your azure openai api key>'
AZURE_OPENAI_ENDPOINT='https://<your endpoint>.openai.azure.com/'
OPENAI_API_VERSION='2024-10-01-preview'

# OpenAI
OPENAI_API_KEY='<your openai api key>'
```


## License

This project is licensed under the MIT License. See the LICENSE file for details.
