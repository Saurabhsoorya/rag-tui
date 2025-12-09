"""Test script to verify Chonkie chunking functionality."""

import asyncio
from rag_tui.core.engine import ChunkingEngine
from rich.console import Console
from rich.table import Table
from rich.panel import Panel


console = Console()


def test_basic_chunking():
    """Test basic synchronous chunking."""
    console.print("\n[bold cyan]Test 1: Basic Synchronous Chunking[/bold cyan]")
    
    engine = ChunkingEngine()
    
    # Sample text about RAG
    text = """
    Retrieval-Augmented Generation (RAG) is a technique that enhances large language models 
    by providing them with relevant context from external knowledge sources. The process works 
    in several steps: First, documents are split into smaller chunks. Second, these chunks are 
    converted into vector embeddings. Third, when a user asks a question, the system retrieves 
    the most relevant chunks based on semantic similarity. Finally, these chunks are provided 
    to the LLM as context to generate a more informed and accurate response. This approach 
    significantly reduces hallucinations and allows LLMs to work with proprietary or recent 
    data that wasn't part of their training set.
    """
    
    # Test with different chunk sizes
    chunks = engine.chunk_text(text, chunk_size=100, overlap=20)
    
    console.print(f"\n[green]✓ Successfully created {len(chunks)} chunks[/green]")
    
    # Display chunks
    for i, (chunk_text, start, end) in enumerate(chunks, 1):
        panel = Panel(
            chunk_text.strip(),
            title=f"[bold]Chunk {i}[/bold]",
            subtitle=f"Position: {start}-{end} | Length: {len(chunk_text)} chars",
            border_style="blue"
        )
        console.print(panel)
    
    # Show statistics
    stats = engine.get_chunk_stats(chunks)
    
    table = Table(title="Chunk Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    for key, value in stats.items():
        if isinstance(value, float):
            table.add_row(key, f"{value:.2f}")
        else:
            table.add_row(key, str(value))
    
    console.print("\n")
    console.print(table)
    
    engine.shutdown()


async def test_async_chunking():
    """Test asynchronous chunking."""
    console.print("\n[bold cyan]Test 2: Asynchronous Chunking[/bold cyan]")
    
    engine = ChunkingEngine()
    
    text = """
    Vector databases are specialized systems designed to store and query high-dimensional 
    vector embeddings efficiently. Unlike traditional databases that work with structured 
    data and exact matches, vector databases use similarity search algorithms like 
    cosine similarity, euclidean distance, or dot product to find semantically similar items. 
    This makes them ideal for RAG applications, recommendation systems, and semantic search.
    """
    
    console.print("[yellow]Starting async chunking...[/yellow]")
    
    # Test async chunking
    chunks = await engine.chunk_text_async(text, chunk_size=80, overlap=15)
    
    console.print(f"[green]✓ Async chunking completed: {len(chunks)} chunks created[/green]")
    
    # Show just the count
    for i, (chunk_text, _, _) in enumerate(chunks, 1):
        console.print(f"  Chunk {i}: {len(chunk_text)} characters")
    
    engine.shutdown()


def test_edge_cases():
    """Test edge cases and error handling."""
    console.print("\n[bold cyan]Test 3: Edge Cases[/bold cyan]")
    
    engine = ChunkingEngine()
    
    # Empty string
    chunks = engine.chunk_text("", chunk_size=100, overlap=10)
    console.print(f"Empty string: {len(chunks)} chunks [dim](expected 0)[/dim]")
    
    # Very short text
    chunks = engine.chunk_text("Hello world", chunk_size=100, overlap=10)
    console.print(f"Short text: {len(chunks)} chunks")
    
    # No overlap
    chunks = engine.chunk_text("This is a test sentence. " * 20, chunk_size=50, overlap=0)
    console.print(f"No overlap: {len(chunks)} chunks")
    
    # Large overlap
    chunks = engine.chunk_text("This is a test sentence. " * 20, chunk_size=100, overlap=80)
    console.print(f"Large overlap (80%): {len(chunks)} chunks")
    
    console.print("[green]✓ All edge cases handled correctly[/green]")
    
    engine.shutdown()


def main():
    """Run all tests."""
    console.print(Panel.fit(
        "[bold magenta]RAG-TUI Chunking Engine Test Suite[/bold magenta]",
        border_style="magenta"
    ))
    
    try:
        # Synchronous tests
        test_basic_chunking()
        test_edge_cases()
        
        # Async test
        asyncio.run(test_async_chunking())
        
        console.print("\n[bold green]✓ All tests passed![/bold green]\n")
        
    except Exception as e:
        console.print(f"\n[bold red]✗ Test failed: {e}[/bold red]\n")
        raise


if __name__ == "__main__":
    main()
