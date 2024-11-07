# MagTest - LLM Benchmark Testing Framework

A framework for testing Large Language Models (LLMs) against various standardized benchmarks.

## Supported Benchmarks

### MMLU (Massive Multitask Language Understanding)

- **Description**: Tests model knowledge across 57 subjects including mathematics, history, law, and medicine
- **Variants**: 0-shot and 5-shot testing
- **Source**: [MMLU Repository](https://github.com/hendrycks/test)
- **Format**: Multiple choice (A, B, C, D)

### GSM8K (Grade School Math 8K)

- **Description**: Tests mathematical reasoning with grade school level word problems
- **Variants**: 0-shot and 4-shot testing
- **Source**: [GSM8K Repository](https://github.com/openai/grade-school-math)
- **Format**: Step-by-step solutions with numerical answers

### BBH (Big-Bench Hard)

- **Description**: Collection of challenging tasks designed to test advanced reasoning capabilities
- **Variants**: 0-shot and 3-shot testing
- **Source**: [BBH Repository](https://github.com/suzgunmirac/BIG-Bench-Hard)
- **Format**: Various formats depending on task type

## Supported Models

### OpenAI Models

- gpt-4o
- gpt-4o-mini
- gpt-4-turbo
- gpt-4
- gpt-3.5-turbo-0125

### Anthropic Models

- claude-3-5-sonnet-20240620
- claude-3-opus-20240229
- claude-3-sonnet-20240229
- claude-3-haiku-20240307

### Google Models

- gemini-1.5-flash-002
- gemini-1.5-flash-001
- gemini-1.5-pro-002
- gemini-1.5-pro-001
- gemini-1.0-pro-002
- gemini-1.0-pro-001

## Features

- Batch and one-by-one testing modes
- Token usage tracking
- Cost estimation
- Performance metrics
- Database storage of results
- Configurable few-shot learning
- Multi-model comparison

## Setup

1. Clone the repository
2. Ensure `venv` is installed:
   - On Windows: `python -m pip install --user virtualenv`
   - On macOS/Linux: `python3 -m pip install --user virtualenv`
3. Create a virtual environment: `python -m venv venv`
4. Activate the virtual environment:
   - On Windows: `venv\Scripts\activate`
   - On macOS/Linux: `source venv/bin/activate`
5. Install dependencies: `pip install -r requirements.txt`
6. Create a `.env` file based on `example.env` and fill in your API keys and configuration details
7. Run tests using Jupyter Notebook: Open `test.ipynb` in Jupyter and execute the cells

## Project Structure
