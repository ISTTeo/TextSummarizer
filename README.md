# Text Summarizer Using Phi 3

This repository contains a text summarizer that leverages the Phi 3 model to generate summaries from textual artifacts. The summarizer can process large texts, generate summaries, and convert them into markdown and PDF formats.

$~$
## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [General Runner Usage](#general-runner-usage)
  - [Example of Usage](#example-of-usage)
- [Configuration](#configuration)
- [Input Data Format](#input-data-format)
- [Output](#output)
- [Contributing](#contributing)



$~$
## Features
- **Multi-GPU Support**: Efficiently utilize multiple GPUs for faster processing.
- **Customizable Commands**: Easily modify default commands and arguments.
- **Markdown Generation**: Convert JSON summaries into well-structured Markdown documents.
- **PDF Conversion**: Transform Markdown files into PDF documents.
- **Organized Summaries**: Automatically organize summaries by author and generate a table of contents.
  
$~$
## Installation
1. Fork the repository.
```bash
git clone https://github.com/your-username/TextSummarizer.git
cd TextSummarizer
```

2. Install the required dependencies:
```bash
pip install torch transformers nltk tqdm
```


3. Download the NLTK punkt tokenizer:
```python
import nltk
nltk.download('punkt')
```

4. install CUDA toolkit and appropriate GPU drivers for your system to enable GPU acceleration.

5. Ensure you have sufficient disk space and RAM to load and run the Phi-3 model.

6. Run the main script `llm_general_multi_process_gpu.py` with the appropriate command-line arguments as shown in the example usage.

$~$
## Usage

### General Runner Usage
To run the summarizer, use the following command:

```bash
python3 utility/llm_general_multi_process_gpu.py \
    --input_json ./youtube/videos.json \
    --output_dir ./youtube/partitions \
    --gpu_devices 5 6 7 \
    --processes_per_gpu 2 \
    --model_id="microsoft/Phi-3.5-mini-instruct" \
    --first_summary_field="phi_mini_summary" \
    --summary_over_summary_field="summary_over_summary" \
    --artifact_text_field="transcript" \
    --debug
```

$~$
### Example of Usage

1. **Generate Markdown from JSON**:

```bash
python3 utility/convert_json_with_summaries_to_md.py --input_json ./youtube/videos.json --output_md ./youtube/videos.md
```

2. **Convert Markdown to PDF**:

```bash
python3 utility/create_md_pdf.py --input_md ./youtube/videos.md --output_pdf ./youtube/videos.pdf
```


3. **Generate Summaries with Custom Arguments**:

```bash
python3 utility/llm_general_multi_process_gpu.py \
    --input_json ./custom_data/input.json \
    --output_dir ./custom_data/output \
    --gpu_devices 0 1 \
    --processes_per_gpu 1 \
    --model_id="microsoft/Phi-3.5-mini-instruct" \
    --first_summary_field="custom_summary" \
    --summary_over_summary_field="custom_summary_over_summary" \
    --artifact_text_field="custom_transcript" \
    --debug
```

$~$
## Configuration

- `default_llm_commands.json`: Contains default instructions for the summarizer. You can create a custom file to override these commands.
- `generation_args.json`: Specifies generation arguments for the language model. Modify this file to adjust generation parameters.

$~$
## Input Data Format
The input JSON file should contain a list of objects, each representing a document to be summarized. Each object should include:
- The artifact's text field (specified by `--artifact_text_field`)
- Any relevant context information

$~$
## Output
The script generates JSON files with summaries in the specified output directory. You can use the utility scripts in the `utility` folder to convert the JSON output to Markdown and PDF formats.


$~$
## Contributing
Contributions to this project are welcome! Please fork the repository and submit a pull request with your proposed changes.
