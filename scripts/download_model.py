"""
scripts/download_model.py
──────────────────────────
Download the Mistral-7B-Instruct GGUF model from Hugging Face Hub.

Usage::

    python scripts/download_model.py
    python scripts/download_model.py --model-id TheBloke/Mistral-7B-Instruct-v0.2-GGUF
                                     --filename mistral-7b-instruct-v0.2.Q4_K_M.gguf
"""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer()

@app.command()
def main(
    model_id: str = typer.Option(
        "TheBloke/Mistral-7B-Instruct-v0.3-GGUF",
        help="Hugging Face repo ID for the GGUF model.",
    ),
    filename: str = typer.Option(
        "mistral-7b-instruct-v0.3.Q4_K_M.gguf",
        help="GGUF filename within the repo.",
    ),
    output_dir: Path = typer.Option(
        Path("models"),
        help="Local directory to save the model.",
    ),
) -> None:
    """Download a GGUF model from Hugging Face Hub."""
    try:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415
    except ImportError:
        typer.echo("huggingface_hub not installed. Run: pip install huggingface-hub")
        raise typer.Exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / filename

    if dest.exists():
        typer.echo(f"Model already present: {dest}")
        raise typer.Exit(0)

    typer.echo(f"Downloading {filename} from {model_id}…")
    path = hf_hub_download(
        repo_id   = model_id,
        filename  = filename,
        local_dir = str(output_dir),
        resume_download=True,
    )
    typer.echo(f"✓ Saved to {path}")


if __name__ == "__main__":
    app()
