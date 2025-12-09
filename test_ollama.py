"""Quick test of Ollama integration with existing models."""

import asyncio
from rag_tui.core.llm import OllamaLLM
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()


async def test_ollama():
    """Test Ollama with existing models."""
    console.print("\n[bold cyan]Testing Ollama Integration[/bold cyan]\n")
    
    # Initialize with your models
    llm = OllamaLLM(model="llama3.2:1b", embedding_model="nomic-embed-text")
    
    # Test 1: Connection
    console.print("[yellow]1. Testing connection to Ollama...[/yellow]")
    connected = await llm.check_connection()
    
    if connected:
        console.print("[green]✓ Connected to Ollama successfully[/green]")
    else:
        console.print("[red]✗ Could not connect to Ollama[/red]")
        return
    
    # Test 2: Embedding
    console.print("\n[yellow]2. Testing embedding generation...[/yellow]")
    try:
        text = "Retrieval-Augmented Generation is powerful"
        embedding = await llm.embed(text)
        console.print(f"[green]✓ Generated embedding with dimension: {len(embedding)}[/green]")
        console.print(f"[dim]First 5 values: {embedding[:5]}[/dim]")
    except Exception as e:
        console.print(f"[red]✗ Embedding failed: {e}[/red]")
        return
    
    # Test 3: Simple generation
    console.print("\n[yellow]3. Testing text generation with llama3.2:1b...[/yellow]")
    try:
        prompt = "Explain what RAG (Retrieval-Augmented Generation) is in one sentence."
        console.print(f"[dim]Prompt: {prompt}[/dim]\n")
        
        response = await llm.generate(prompt, temperature=0.7)
        
        panel = Panel(
            response,
            title="[bold]LLM Response[/bold]",
            border_style="green"
        )
        console.print(panel)
    except Exception as e:
        console.print(f"[red]✗ Generation failed: {e}[/red]")
        return
    
    # Test 4: RAG prompt building
    console.print("\n[yellow]4. Testing RAG prompt building...[/yellow]")
    
    context_chunks = [
        "RAG combines retrieval with generation.",
        "It reduces hallucinations in LLMs.",
        "Vector databases store embeddings for similarity search."
    ]
    
    rag_prompt = llm.build_rag_prompt(
        "What are the benefits of RAG?",
        context_chunks
    )
    
    console.print("[green]✓ RAG prompt built successfully[/green]")
    console.print(f"[dim]Prompt length: {len(rag_prompt)} characters[/dim]")
    
    console.print("\n[bold green]✓ All Ollama tests passed![/bold green]")
    console.print("\n[bold cyan]Your models are ready to use:[/bold cyan]")
    console.print("  • LLM: llama3.2:1b")
    console.print("  • Embeddings: nomic-embed-text\n")


async def main():
    """Run tests."""
    try:
        await test_ollama()
    except Exception as e:
        console.print(f"\n[bold red]✗ Test failed: {e}[/bold red]\n")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
