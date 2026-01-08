#!/usr/bin/env python3
"""
Interactive CLI Chat with Human-in-the-Loop RLM

A conversational interface that always starts with human_query() to ask what you want to accomplish.
"""

import os
import sys
from dotenv import load_dotenv

from rlm import RLM

load_dotenv()


def load_context_files(file_paths_str: str) -> dict | None:
    """Load multiple context files from comma/space-separated paths.
    
    Supports both individual files and directories (non-recursive).
    For directories, loads all non-hidden files from the root level only.
    
    Returns:
        dict with file contents or None if skipped/error
    """
    if not file_paths_str.strip():
        return None
    
    # Split by comma or space
    paths = [p.strip() for p in file_paths_str.replace(',', ' ').split() if p.strip()]
    
    context_data = {}
    errors = []
    
    for path in paths:
        # Check if path is a directory
        if os.path.isdir(path):
            try:
                # List all non-hidden files in directory (non-recursive)
                files_in_dir = [
                    os.path.join(path, f) 
                    for f in os.listdir(path) 
                    if not f.startswith('.') and os.path.isfile(os.path.join(path, f))
                ]
                
                if not files_in_dir:
                    errors.append(f"No readable files in directory: {path}")
                    continue
                
                # Try to read all files - if any fail, fail the entire directory
                dir_contents = {}
                dir_errors = []
                for file_path in files_in_dir:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            dir_contents[file_path] = content
                    except UnicodeDecodeError:
                        dir_errors.append(f"Cannot read as text (binary?): {file_path}")
                    except Exception as e:
                        dir_errors.append(f"Error reading {file_path}: {e}")
                
                # Strict rule: if any file in directory failed, fail the whole directory
                if dir_errors:
                    errors.append(f"Failed to load directory '{path}' (strict mode):")
                    errors.extend([f"  - {e}" for e in dir_errors])
                else:
                    # All files loaded successfully
                    context_data.update(dir_contents)
                    
            except PermissionError:
                errors.append(f"Permission denied for directory: {path}")
            except Exception as e:
                errors.append(f"Error reading directory {path}: {e}")
        else:
            # Regular file
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    context_data[path] = content
            except FileNotFoundError:
                errors.append(f"File not found: {path}")
            except UnicodeDecodeError:
                errors.append(f"Cannot read as text (binary file?): {path}")
            except PermissionError:
                errors.append(f"Permission denied: {path}")
            except Exception as e:
                errors.append(f"Error reading {path}: {e}")
    
    if errors:
        print("\n❌ Errors loading files:")
        for error in errors:
            print(f"   {error}")
        return None
    
    return context_data if context_data else None


def main():
    print("🤖 Human-in-the-Loop RLM Chat")
    print("=" * 50)
    print("Type 'quit' or 'exit' to end the conversation")
    print("💡 Tip: Set RLM_VERBOSE=true to see detailed progress")
    print()

    # Initialize RLM with human_query enabled
    # Set verbose=True to see detailed progress during development
    verbose_mode = os.getenv("RLM_VERBOSE", "false").lower() == "true"
    
    # Set up HumanHandler for human_query support
    from rlm.core.human_handler import HumanHandler
    human_handler = HumanHandler()
    human_handler_address = (human_handler.host, human_handler.port)
    
    rlm = RLM(
        backend="openai",
        backend_kwargs={
            "model_name": "anthropic/claude-3.5-sonnet",  # Back to Claude for testing
            "api_key": os.getenv("OPENROUTER_API_KEY"),
            "base_url": "https://openrouter.ai/api/v1",
            "max_tokens": 2000,  # Good response length
        },
        environment_kwargs={
            "human_handler_address": human_handler_address,
        },
        max_iterations=10,  # Increased to 10 for complex queries
        verbose=verbose_mode,  # Enable with RLM_VERBOSE=true environment variable
    )
    
    if verbose_mode:
        print("🔍 Verbose mode enabled")
        print()

    while True:
        try:
            # Prompt for context files (optional)
            print("📁 Context files/folders (optional): Enter paths separated by commas or spaces")
            print("   Examples: data.txt, notes.md  OR  /path/to/folder  OR  file.json, ./my_dir")
            print("   Note: Folders load all non-hidden files (non-recursive)")
            print("   Press Enter to skip")
            print("> ", end="", flush=True)
            context_files_input = input().strip()
            
            # Allow user to quit at context prompt
            if context_files_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            # Try to load context files
            context_files = None
            if context_files_input:
                context_files = load_context_files(context_files_input)
                if context_files:
                    total_chars = sum(len(content) for content in context_files.values())
                    print(f"✓ Loaded {len(context_files)} file(s) ({total_chars:,} characters)")
                    for path in context_files.keys():
                        print(f"  - {path}")
                    print()
                else:
                    # Files failed to load - ask if they want to retry
                    retry = input("Try again with different files? (y/n): ").lower().strip()
                    if retry in ['y', 'yes']:
                        continue
                    # Otherwise proceed without context
                    print("Continuing without context files...\n")
            
            # Get user task
            print("What would you like help with? (or 'quit'/'exit' to end)")
            print("> ", end="", flush=True)
            user_task = input().strip()
            
            if user_task.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
                
            if not user_task:
                print("Please enter a task or question.")
                continue

            # Build context with or without files
            # Flatten structure so QueryMetadata can properly calculate sizes
            if context_files:
                context = {
                    "user_task": user_task,
                    **context_files  # Flatten files into the context dict
                }
            else:
                context = {"user_task": user_task}
            
            # Root prompt tells RLM what the goal is
            root_prompt = f"Help the user with: {user_task}"

            if not verbose_mode:
                print("\n🤔 Processing your request...")
                print("⏳ The AI may ask you follow-up questions using human_query()...")
                print()
            
            result = rlm.completion(context, root_prompt=root_prompt)

            # Only print final answer if not in verbose mode (verbose mode already shows it)
            if not verbose_mode:
                print(f"\n{'='*50}")
                print(f"🎯 Final Answer:\n{result.response}")
                print(f"{'='*50}\n")
            else:
                print()  # Just add some spacing

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            choice = input("Try again? (y/n): ").lower().strip()
            if choice not in ['y', 'yes']:
                break


if __name__ == "__main__":
    main()
