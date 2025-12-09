"""Test script for the complete RAG-TUI application (without UI)."""

import asyncio
import numpy as np
from rag_tui.core.engine import ChunkingEngine
from rag_tui.core.vector import VectorStore, EmbeddingCache
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


async def test_integration():
    """Test the integration of all core components."""
    console.print("\n[bold cyan]RAG-TUI Integration Test[/bold cyan]\n")
    
    # Sample text
    text = """Retrieval-Augmented Generation (RAG) is a powerful technique. It combines 
    the capabilities of large language models with external knowledge retrieval. This approach 
    helps reduce hallucinations and provides more accurate, grounded responses."""
    
    # Test 1: Chunking
    console.print("[yellow]1. Testing Chunking Engine...[/yellow]")
    engine = ChunkingEngine()
    chunks = await engine.chunk_text_async(text, chunk_size=50, overlap=10)
    console.print(f"[green]✓ Created {len(chunks)} chunks[/green]")
    
    # Test 2: Vector Store
    console.print("\n[yellow]2. Testing Vector Store...[/yellow]")
    store = VectorStore(embedding_dim=768)
    
    # Create dummy embeddings
    chunk_texts = [c[0] for c in chunks]
    dummy_embeddings = np.random.randn(len(chunk_texts), 768).astype(np.float32)
    
    store.add_chunks(chunk_texts, dummy_embeddings)
    console.print(f"[green]✓ Added {len(chunk_texts)} chunks to vector store[/green]")
    
    # Test 3: Search
    console.print("\n[yellow]3. Testing Search...[/yellow]")
    query_embedding = np.random.randn(768).astype(np.float32)
    results = await store.search_async(query_embedding, top_k=2)
    
    console.print(f"[green]✓ Retrieved {len(results)} results[/green]")
    
    # Display results
    table = Table(title="Search Results")
    table.add_column("Rank", style="cyan")
    table.add_column("Score", style="magenta")
    table.add_column("Chunk Preview", style="white")
    
    for i, (chunk, score, _) in enumerate(results, 1):
        preview = chunk[:50] + "..." if len(chunk) > 50 else chunk
        table.add_row(str(i), f"{score:.3f}", preview)
    
    console.print("\n")
    console.print(table)
    
    # Test 4: Stats
    console.print("\n[yellow]4. Testing Statistics...[/yellow]")
    chunk_stats = engine.get_chunk_stats(chunks)
    store_stats = store.get_stats()
    
    stats_panel = Panel(
        f"""Chunking Stats:
  - Total Chunks: {chunk_stats['total_chunks']}
  - Avg Size: {chunk_stats['avg_chunk_size']:.1f} chars
  
Vector Store Stats:
  - Total Indexed: {store_stats['total_chunks']}
  - Embedding Dim: {store_stats['embedding_dim']}
  - Memory Usage: {store_stats['memory_usage_mb']:.2f} MB""",
        title="[bold]System Statistics[/bold]",
        border_style="green"
    )
    console.print(stats_panel)
    
    # Cleanup
    engine.shutdown()
    store.shutdown()
    
    console.print("\n[bold green]✓ All integration tests passed![/bold green]\n")


async def main():
    """Run integration tests."""
    try:
        await test_integration()
    except Exception as e:
        console.print(f"\n[bold red]✗ Test failed: {e}[/bold red]\n")
        raise


if __name__ == "__main__":
    asyncio.run(main())
