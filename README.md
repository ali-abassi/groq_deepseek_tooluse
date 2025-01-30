# Groq Tool Use Example

This repository demonstrates how to implement tool use with the Groq API, specifically focusing on CSV and file handling operations in a sequential manner.

## Overview

The `Groq_Tool_Use.py` implementation showcases how to create a chat interface that can:
- Handle file operations (read/write)
- Manage CSV files with various operations
- Process tool calls sequentially
- Provide interactive responses with rich formatting

## Features

### File Operations
- Read file contents
- Write content to files

### CSV Operations
- Create new CSV files with headers and data
- Read CSV file contents
- Append rows to existing CSV files
- Update specific cells in CSV files
- Query CSV files using pandas syntax
- Add new columns to existing CSV files

### Interactive Features
- Rich console output with formatted panels
- Markdown rendering support
- Progress indicators during processing
- Error handling with visual feedback
- Conversation history management

## Requirements

```python
from groq import Groq
import os
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich import box
from rich.markdown import Markdown
import json
import re
import pandas as pd
import csv
```

## Setup

1. Install required packages:
```bash
pip install groq python-dotenv rich pandas
```

2. Change .envsample to a `.env` file with your Groq API key:
```
GROQ_API_KEY=your_api_key_here
```

3. Run the script:
```bash
python Groq_Tool_Use.py
```

## Tool Implementation

The implementation follows a sequential tool calling pattern where:
1. User input is received
2. Initial API call is made to Groq
3. Tool calls are processed one at a time
4. Results are collected and formatted
5. Final response is generated

### Available Tools

1. `read_file_tool`: Read contents of a file
2. `write_file_tool`: Write content to a file
3. `create_csv_tool`: Create a new CSV file with headers and data
4. `read_csv_tool`: Read CSV file contents
5. `append_csv_tool`: Append rows to existing CSV
6. `update_csv_tool`: Update specific cells in CSV
7. `query_csv_tool`: Query CSV using pandas syntax
8. `add_columns_csv_tool`: Add new columns to CSV

## Usage Example

```python
# Initialize the chat interface
chat = GroqDeepseek()

# Create a CSV file
chat.chat("Create a CSV file with columns for name, age, and city")

# Add data to the CSV
chat.chat("Add a new row with data: John, 30, New York")

# Query the CSV
chat.chat("Show me all entries where age is greater than 25")
```

## Error Handling

The implementation includes comprehensive error handling:
- Tool execution errors are caught and formatted
- API call errors are handled gracefully
- File operation errors are reported with clear messages
- CSV operation errors include detailed feedback

## Limitations

- Tools are called sequentially, not in parallel
- Each tool call requires a separate API interaction
- Response formatting is tied to the Rich library
- CSV operations are memory-bound for large files

## Contributing

Feel free to contribute by:
1. Opening issues for bugs or feature requests
2. Submitting pull requests with improvements
3. Adding documentation or examples
4. Suggesting new tool implementations

## License

MIT License - Feel free to use and modify as needed. 
