"""
Documentation Generator - Docs-as-Code Tool

Analyzes git diffs and generates documentation updates using AI providers.
Supports OpenAI (default) and DeepSeek.
"""

import os
import subprocess
import sys

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# --- CONFIGURATION MAPPING ---
PROVIDERS = {
    "openai": {
        "env_var": "OPENAI_API_KEY",
        "base_url": None,  # Defaults to OpenAI's standard URL
        "model": "gpt-4-turbo",
        "name": "OpenAI GPT-4 Turbo"
    },
    "deepseek": {
        "env_var": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
        "name": "DeepSeek V3"
    }
}

# --- THE MASTER PROMPT ---
SYSTEM_PROMPT = """
You are the Lead Documentation Architect.
Your goal: Analyze code changes and generate Markdown documentation updates.

RULES:
1. Analyze the GIT DIFF provided.
2. Structure output for a standard Docs-as-Code hierarchy:
   - architecture/ (decisions, context)
   - design/ (data-model, security)
   - technical/ (api, setup)
   - operations/ (config, deploy)
3. Output format:
   - Start with "### FILE: [Relative Path]" (e.g., "04-operations/configuration.md")
   - Provide the Markdown content to append or replace.
   - Use Mermaid.js for diagrams.
4. If trivial (typo, formatting), output "NO_UPDATES".
"""


def get_git_diff(repo_path: str) -> str:
    """Get the git diff between the last two commits.

    Args:
        repo_path: Path to the git repository.

    Returns:
        The git diff output as a string.

    Raises:
        SystemExit: If git command fails or git is not installed.
    """
    try:
        result = subprocess.run(
            ["git", "diff", "HEAD~1", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error running git in {repo_path}: {e}")
    except FileNotFoundError:
        raise RuntimeError("Git is not installed.")


def get_staged_diff(repo_path: str) -> str:
    """Get the git diff of staged changes.

    Args:
        repo_path: Path to the git repository.

    Returns:
        The git diff output as a string.

    Raises:
        RuntimeError: If git command fails or git is not installed.
    """
    try:
        result = subprocess.run(
            ["git", "diff", "--cached"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error running git in {repo_path}: {e}")
    except FileNotFoundError:
        raise RuntimeError("Git is not installed.")


def get_unstaged_diff(repo_path: str) -> str:
    """Get the git diff of unstaged changes.

    Args:
        repo_path: Path to the git repository.

    Returns:
        The git diff output as a string.

    Raises:
        RuntimeError: If git command fails or git is not installed.
    """
    try:
        result = subprocess.run(
            ["git", "diff"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error running git in {repo_path}: {e}")
    except FileNotFoundError:
        raise RuntimeError("Git is not installed.")


def get_commit_diff(repo_path: str, commit_id: str = "HEAD") -> str:
    """Get the git diff for a specific commit.

    Args:
        repo_path: Path to the git repository.
        commit_id: Commit SHA or reference (default: HEAD).

    Returns:
        The git diff output as a string.

    Raises:
        RuntimeError: If git command fails or git is not installed.
    """
    try:
        result = subprocess.run(
            ["git", "show", "--format=", commit_id],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error getting commit {commit_id}: {e.stderr.strip() if e.stderr else e}")
    except FileNotFoundError:
        raise RuntimeError("Git is not installed.")


def generate_docs(diff_content: str, provider_key: str) -> str:
    """Generate documentation using the specified AI provider.

    Args:
        diff_content: The git diff to analyze.
        provider_key: The AI provider to use ('openai' or 'deepseek').

    Returns:
        The generated documentation as a string.

    Raises:
        ValueError: If API key is not configured.
        RuntimeError: If API call fails.
    """
    # Load configuration based on the chosen provider
    config = PROVIDERS[provider_key]
    api_key = os.getenv(config["env_var"])

    if not api_key:
        raise ValueError(f"{config['env_var']} environment variable not found.")

    # Initialize Client
    # Note: If base_url is None, the library defaults to OpenAI
    client = OpenAI(api_key=api_key, base_url=config["base_url"])

    try:
        response = client.chat.completions.create(
            model=config["model"],
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Here is the Git Diff:\n\n{diff_content}"}
            ],
            temperature=0.2,
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"Error calling API: {e}")


def run_docs_generator(
    repo_path: str = ".",
    output_dir: str = None,
    provider: str = "openai",
    diff_source: str = "commit",
    commit_id: str = None
) -> tuple[bool, str]:
    """Run the documentation generator.

    Args:
        repo_path: Path to the git repository.
        output_dir: Directory to save the output. Defaults to <repo>/docs.
        provider: AI provider to use ('openai' or 'deepseek').
        diff_source: Source of diff ('commit', 'staged', 'unstaged', 'all',
                     'current_commit', 'specific_commit').
        commit_id: Specific commit ID when diff_source is 'specific_commit'.

    Returns:
        Tuple of (success: bool, message: str).
    """
    # Resolve Paths
    repo_path = os.path.abspath(os.path.expanduser(repo_path))
    if output_dir:
        output_dir = os.path.abspath(os.path.expanduser(output_dir))
    else:
        output_dir = os.path.join(repo_path, "docs")

    # Get Changes based on diff_source
    try:
        if diff_source == "commit":
            diff = get_git_diff(repo_path)
        elif diff_source == "staged":
            diff = get_staged_diff(repo_path)
        elif diff_source == "unstaged":
            diff = get_unstaged_diff(repo_path)
        elif diff_source == "all":
            # Combine staged and unstaged
            staged = get_staged_diff(repo_path)
            unstaged = get_unstaged_diff(repo_path)
            diff = staged + "\n" + unstaged
        elif diff_source == "current_commit":
            diff = get_commit_diff(repo_path, "HEAD")
        elif diff_source == "specific_commit":
            if not commit_id:
                return False, "No commit ID provided for specific_commit source."
            diff = get_commit_diff(repo_path, commit_id)
        else:
            return False, f"Invalid diff_source: {diff_source}"
    except RuntimeError as e:
        return False, str(e)

    if not diff.strip():
        return False, "No changes found."

    # Safety Truncate (DeepSeek handles larger contexts better)
    limit = 30000 if provider == "deepseek" else 15000
    truncated = False
    if len(diff) > limit:
        diff = diff[:limit]
        truncated = True

    # Generate Content
    try:
        docs_update = generate_docs(diff, provider)
    except (ValueError, RuntimeError) as e:
        return False, str(e)

    # Save Output
    if "NO_UPDATES" in docs_update:
        return True, "No documentation updates required."

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "docs_suggestion.md")

    with open(output_file, "w") as f:
        f.write(docs_update)

    message = f"Documentation saved to: {output_file}"
    if truncated:
        message = f"Warning: Diff was truncated to {limit} chars.\n{message}"

    return True, message


def main():
    """Main entry point for standalone execution."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate Docs-as-Code updates from git changes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Use OpenAI (default) on current repo
  %(prog)s --provider deepseek      # Use DeepSeek instead
  %(prog)s -p deepseek -r /my/repo  # DeepSeek on specific repo
  %(prog)s -o ./my-docs             # Custom output directory
        """
    )

    # Path Arguments
    parser.add_argument(
        "--repo", "-r",
        default=".",
        help="Path to the project root (default: current directory)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Directory to save the output (default: <repo>/docs)"
    )

    # Provider Argument
    parser.add_argument(
        "--provider", "-p",
        choices=list(PROVIDERS.keys()),
        default="openai",
        help="Choose the AI provider (default: openai)"
    )

    # Diff source argument
    parser.add_argument(
        "--diff-source", "-s",
        choices=["commit", "staged", "unstaged", "all"],
        default="commit",
        help="Source of changes to analyze (default: commit)"
    )

    args = parser.parse_args()

    print(f"Project Path:    {os.path.abspath(args.repo)}")
    print(f"Using Provider:  {PROVIDERS[args.provider]['name']}")
    print(f"Diff Source:     {args.diff_source}")
    print()
    print("AI is analyzing changes...")

    success, message = run_docs_generator(
        repo_path=args.repo,
        output_dir=args.output,
        provider=args.provider,
        diff_source=args.diff_source
    )

    if success:
        print()
        print("=" * 50)
        print(message)
        print("=" * 50)
    else:
        print(f"Error: {message}")
        sys.exit(1)


if __name__ == "__main__":
    main()
