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

2. Install the package locally (registers the `pycodei` command and all dependencies):
    ```bash
    pip install -e .
    ```

3. Create your user configuration file:
    ```bash
    mkdir -p ~/.pycodei
    cat > ~/.pycodei/config.json <<'EOF'
    {
      "DEPLOYMENT_NAME": "gpt-4o-mini",
      "AZURE_OPENAI_API_KEY": "<your azure openai api key>",
      "AZURE_OPENAI_ENDPOINT": "https://<your endpoint>.openai.azure.com/",
      "OPENAI_API_VERSION": "2024-10-01-preview",
      "OPENAI_API_KEY": "<your openai api key>"
    }
    EOF
    ```
    Running `pycodei --help` will also create this file with placeholder values if it does not exist.
4. Set up data directory:
    ```bash
    sudo ln -s `pwd`/sample_data /mnt/data
    ```

## Usage
To use the Python Code Interpreter, run the following command (replace the prompt text with your task):
```bash
pycodei "Retrieve NVIDIA stock prices from Jan-Mar 2023 and predict April onward."
```
Use `pycodei --help` to see optional flags such as `--deployment-name`, which overrides the value in `~/.pycodei/config.json`.

### Example
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


#### Example 1
"Retrieve NVIDIA stock prices from Yahoo for January to March 2023 and predict prices from April 2023 onwards."  
Result: [sample_01.ipynb](https://github.com/KentaroAOKI/python_code_interpreter/blob/main/sample_results/sample_01.ipynb)

#### Example 2
"Determine whether the data in /mnt/data/diagnosis.csv is malignant or benign. To make a decision, use the model learned using the load_breast_cancer data available from scikit-learn."  
Result: [sample_02.ipynb](https://github.com/KentaroAOKI/python_code_interpreter/blob/main/sample_results/sample_02.ipynb)

#### Example 3
"2023年の1月から3月のNVIDIA株価をYahooから取得して、2023年4月以降の株価を予測してください。"  
Result: [sample_03.ipynb](https://github.com/KentaroAOKI/python_code_interpreter/blob/main/sample_results/sample_03.ipynb)

#### Example 4
"/mnt/data/diagnosis.csv のデータが悪性か良性か判断してください。判断は、scikit-learn から取得できる load_breast_cancer データで学習したモデルを使ってください。日本語で説明してください。"  
Result: [sample_04.ipynb](https://github.com/KentaroAOKI/python_code_interpreter/blob/main/sample_results/sample_04.ipynb)

## Configuration
Runtime credentials are read from `~/.pycodei/config.json`. Example:

```json
{
  "DEPLOYMENT_NAME": "gpt-4o-mini",
  "AZURE_OPENAI_API_KEY": "<your azure openai api key>",
  "AZURE_OPENAI_ENDPOINT": "https://<your endpoint>.openai.azure.com/",
  "OPENAI_API_VERSION": "2024-10-01-preview",
  "OPENAI_API_KEY": "<your openai api key>"
}
```
All keys map directly to the environment variables expected by the OpenAI/Azure SDKs. Leaving a value blank may cause authentication failures, so be sure to populate the entries relevant to your deployment.


## License

This project is licensed under the MIT License. See the LICENSE file for details.
