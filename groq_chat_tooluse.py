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
from typing import List, Dict

# Load environment variables
load_dotenv()

console = Console()

def extract_think_content(text):
    """Extract content between <think> tags and format the rest"""
    think_pattern = r'<think>(.*?)</think>'
    matches = re.findall(think_pattern, text, re.DOTALL)
    if matches:
        thinking = matches[0].strip()
        response = re.sub(think_pattern, '', text, flags=re.DOTALL).strip()
        return thinking, response
    return None, text

class GroqDeepseek:
    def __init__(self):
        self.client = Groq(api_key=os.getenv('GROQ_API_KEY'))
        self.conversation_history = []
        self.model = "deepseek-r1-distill-llama-70b"
        
    def read_file_tool(self, file_path: str) -> dict:
        """Tool to read file contents"""
        try:
            with open(file_path, 'r') as file:
                content = file.read()
                return json.dumps({"content": content})
        except Exception as e:
            return json.dumps({"error": f"Error reading file: {str(e)}"})
            
    def write_file_tool(self, file_path: str, content: str) -> dict:
        """Tool to write content to a file"""
        try:
            with open(file_path, 'w') as file:
                file.write(content)
                return json.dumps({"success": True, "message": f"Successfully wrote to {file_path}"})
        except Exception as e:
            return json.dumps({"error": f"Error writing file: {str(e)}"})

    def create_csv_tool(self, file_path: str, headers: List[str], data: List[List[str]]) -> dict:
        """Tool to create a new CSV file with headers and data"""
        try:
            with open(file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(headers)
                writer.writerows(data)
            return json.dumps({"success": True, "message": f"Successfully created CSV file at {file_path}"})
        except Exception as e:
            return json.dumps({"error": f"Error creating CSV file: {str(e)}"})

    def read_csv_tool(self, file_path: str, num_rows: int = None) -> dict:
        """Tool to read CSV file contents"""
        try:
            df = pd.read_csv(file_path)
            if num_rows:
                df = df.head(num_rows)
            return json.dumps({
                "headers": df.columns.tolist(),
                "data": df.values.tolist(),
                "shape": df.shape,
                "preview": df.to_string()
            })
        except Exception as e:
            return json.dumps({"error": f"Error reading CSV file: {str(e)}"})

    def append_csv_tool(self, file_path: str, data: List[List[str]]) -> dict:
        """Tool to append rows to an existing CSV file"""
        try:
            with open(file_path, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(data)
            return json.dumps({"success": True, "message": f"Successfully appended {len(data)} rows to {file_path}"})
        except Exception as e:
            return json.dumps({"error": f"Error appending to CSV file: {str(e)}"})

    def update_csv_tool(self, file_path: str, row_index: int, column_name: str, new_value: str) -> dict:
        """Tool to update a specific cell in a CSV file"""
        try:
            df = pd.read_csv(file_path)
            if row_index >= len(df):
                return json.dumps({"error": f"Row index {row_index} out of bounds"})
            if column_name not in df.columns:
                return json.dumps({"error": f"Column '{column_name}' not found"})
            df.at[row_index, column_name] = new_value
            df.to_csv(file_path, index=False)
            return json.dumps({"success": True, "message": f"Successfully updated cell at row {row_index}, column '{column_name}'"})
        except Exception as e:
            return json.dumps({"error": f"Error updating CSV file: {str(e)}"})

    def query_csv_tool(self, file_path: str, query: str) -> dict:
        """Tool to query CSV file using pandas query syntax"""
        try:
            df = pd.read_csv(file_path)
            result = df.query(query)
            return json.dumps({
                "data": result.values.tolist(),
                "shape": result.shape,
                "preview": result.to_string()
            })
        except Exception as e:
            return json.dumps({"error": f"Error querying CSV file: {str(e)}"})

    def add_columns_csv_tool(self, file_path: str, new_columns: Dict[str, List[str]]) -> dict:
        """Tool to add new columns to an existing CSV file"""
        try:
            # Read existing CSV
            df = pd.read_csv(file_path)
            
            # Add each new column
            for column_name, column_data in new_columns.items():
                if len(column_data) != len(df):
                    return json.dumps({"error": f"Column {column_name} data length ({len(column_data)}) does not match CSV length ({len(df)})"})
                df[column_name] = column_data
            
            # Save back to CSV
            df.to_csv(file_path, index=False)
            return json.dumps({
                "success": True, 
                "message": f"Successfully added columns: {', '.join(new_columns.keys())}",
                "preview": df.to_string()
            })
        except Exception as e:
            return json.dumps({"error": f"Error adding columns to CSV file: {str(e)}"})

    def format_message_for_history(self, message):
        """Format message object for conversation history"""
        if hasattr(message, 'tool_calls'):
            return {
                "role": message.role,
                "content": message.content if message.content else "",
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    }
                    for tool_call in message.tool_calls
                ] if message.tool_calls else []
            }
        return {
            "role": message.role,
            "content": message.content if message.content else ""
        }

    def chat(self, user_input: str):
        # Define available tools
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read contents of a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the file to read"
                            }
                        },
                        "required": ["file_path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write content to a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the file to write"
                            },
                            "content": {
                                "type": "string",
                                "description": "Content to write to the file"
                            }
                        },
                        "required": ["file_path", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_csv",
                    "description": "Create a new CSV file with headers and data",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the CSV file to create"
                            },
                            "headers": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of column headers"
                            },
                            "data": {
                                "type": "array",
                                "items": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                },
                                "description": "List of rows, where each row is a list of values"
                            }
                        },
                        "required": ["file_path", "headers", "data"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "read_csv",
                    "description": "Read contents of a CSV file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the CSV file to read"
                            },
                            "num_rows": {
                                "type": "integer",
                                "description": "Optional: Number of rows to read (reads all if not specified)"
                            }
                        },
                        "required": ["file_path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "append_csv",
                    "description": "Append rows to an existing CSV file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the CSV file"
                            },
                            "data": {
                                "type": "array",
                                "items": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                },
                                "description": "List of rows to append, where each row is a list of values"
                            }
                        },
                        "required": ["file_path", "data"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "update_csv",
                    "description": "Update a specific cell in a CSV file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the CSV file"
                            },
                            "row_index": {
                                "type": "integer",
                                "description": "Index of the row to update (0-based)"
                            },
                            "column_name": {
                                "type": "string",
                                "description": "Name of the column to update"
                            },
                            "new_value": {
                                "type": "string",
                                "description": "New value to set"
                            }
                        },
                        "required": ["file_path", "row_index", "column_name", "new_value"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "query_csv",
                    "description": "Query CSV file using pandas query syntax",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the CSV file"
                            },
                            "query": {
                                "type": "string",
                                "description": "Query string using pandas query syntax (e.g., 'age > 25 and city == \"New York\"')"
                            }
                        },
                        "required": ["file_path", "query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "add_columns_csv",
                    "description": "Add new columns to an existing CSV file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the CSV file"
                            },
                            "new_columns": {
                                "type": "object",
                                "description": "Dictionary of column names and their data",
                                "additionalProperties": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                }
                            }
                        },
                        "required": ["file_path", "new_columns"]
                    }
                }
            }
        ]

        # Add user message to conversation history
        self.conversation_history.append({"role": "user", "content": user_input})

        try:
            # Make initial API call
            with console.status("[bold yellow]Thinking...", spinner="dots"):
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.conversation_history,
                    tools=tools,
                    tool_choice="auto",
                    temperature=0.7,
                    max_completion_tokens=4096
                )

            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls

            if tool_calls:
                # Add assistant's response to conversation
                self.conversation_history.append(self.format_message_for_history(response_message))

                # Process each tool call
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    # Show tool execution in a subtle way
                    console.print(f"[dim]Executing {function_name}...[/]")

                    # Execute the appropriate tool
                    if function_name == "read_file":
                        function_response = self.read_file_tool(**function_args)
                    elif function_name == "write_file":
                        function_response = self.write_file_tool(**function_args)
                    elif function_name == "create_csv":
                        function_response = self.create_csv_tool(**function_args)
                    elif function_name == "read_csv":
                        function_response = self.read_csv_tool(**function_args)
                    elif function_name == "append_csv":
                        function_response = self.append_csv_tool(**function_args)
                    elif function_name == "update_csv":
                        function_response = self.update_csv_tool(**function_args)
                    elif function_name == "query_csv":
                        function_response = self.query_csv_tool(**function_args)
                    elif function_name == "add_columns_csv":
                        function_response = self.add_columns_csv_tool(**function_args)

                    # Add tool response to conversation
                    self.conversation_history.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": function_response
                    })

                # Make second API call with tool results
                with console.status("[bold yellow]Processing...", spinner="dots"):
                    second_response = self.client.chat.completions.create(
                        model=self.model,
                        messages=self.conversation_history,
                        temperature=0.7,
                        max_completion_tokens=4096
                    )
                
                content = second_response.choices[0].message.content
                thinking, response = extract_think_content(content)
                
                if thinking:
                    console.print(Panel(
                        thinking,
                        title="[bold yellow]Thinking Process",
                        border_style="yellow",
                        expand=False,
                        padding=(1, 2)
                    ))
                
                if response:
                    # Try to parse as markdown
                    try:
                        console.print(Markdown(response))
                    except:
                        console.print(response)
                
                return response

            content = response_message.content
            thinking, response = extract_think_content(content)
            
            if thinking:
                console.print(Panel(
                    thinking,
                    title="[bold yellow]Thinking Process",
                    border_style="yellow",
                    expand=False,
                    padding=(1, 2)
                ))
            
            if response:
                # Try to parse as markdown
                try:
                    console.print(Markdown(response))
                except:
                    console.print(response)
            
            return response

        except Exception as e:
            console.print(Panel(
                f"[red]Error: {str(e)}[/]",
                title="Error",
                border_style="red"
            ))
            return f"An error occurred: {str(e)}"

def main():
    chat = GroqDeepseek()
    console.print(Panel(
        "[green]GroqDeepseek initialized! Using the Deepseek model with file read/write capabilities.[/]\n"
        "Type 'quit' to exit.",
        title="[bold]GroqDeepseek[/]",
        border_style="blue",
        padding=(1, 2)
    ))

    while True:
        user_input = console.input("\n[bold green]You:[/] ").strip()

        if user_input.lower() == 'quit':
            console.print(Panel(
                "[yellow]Thank you for using GroqDeepseek! Goodbye![/]",
                border_style="yellow",
                padding=(1, 1)
            ))
            break

        response = chat.chat(user_input)

if __name__ == "__main__":
    main() 
