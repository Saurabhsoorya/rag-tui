"""Chunk card widget for displaying individual chunks with colors."""

from textual.widgets import Static
from textual.containers import Vertical, Container, VerticalScroll
from textual.app import ComposeResult


# Beautiful color palette for chunk borders and backgrounds
CHUNK_COLORS = [
    ("#3b82f6", "#1e3a5f"),  # Blue
    ("#22c55e", "#166534"),  # Green  
    ("#a855f7", "#581c87"),  # Purple
    ("#f97316", "#9a3412"),  # Orange
    ("#06b6d4", "#155e75"),  # Cyan
    ("#ec4899", "#831843"),  # Pink
]


class ChunkCard(Container):
    """A styled card displaying a single chunk with colored border."""
    
    DEFAULT_CSS = """
    ChunkCard {
        height: auto;
        max-height: 20;
        margin: 1 0;
        padding: 1 2;
        border: solid $primary;
        background: $surface;
    }
    
    ChunkCard > Static {
        height: auto;
    }
    
    ChunkCard .chunk-header {
        text-style: bold;
        margin-bottom: 1;
    }
    
    ChunkCard .chunk-scroll {
        height: auto;
        max-height: 12;
        overflow-y: auto;
        padding: 0 1;
        margin-bottom: 1;
    }
    
    ChunkCard .chunk-content {
        height: auto;
    }
    
    ChunkCard .chunk-meta {
        color: $text-muted;
        text-style: italic;
    }
    """
    
    def __init__(
        self,
        chunk_text: str,
        chunk_index: int,
        start_pos: int,
        end_pos: int,
        token_count: int = 0,
        **kwargs
    ):
        """Initialize the chunk card."""
        super().__init__(**kwargs)
        self.chunk_text = chunk_text
        self.chunk_index = chunk_index
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.token_count = token_count
        
        # Get color for this chunk
        color_idx = chunk_index % len(CHUNK_COLORS)
        self.border_color, self.bg_color = CHUNK_COLORS[color_idx]
    
    def _get_quality_indicators(self) -> str:
        """Analyze chunk quality and return indicators."""
        indicators = []
        text = self.chunk_text.strip()
        
        # Check if ends with sentence terminator
        if text and text[-1] in '.!?':
            indicators.append("🟢")
        elif text and text[-1] in ',:;':
            indicators.append("🟡")  # Ends mid-phrase
        else:
            indicators.append("🔴")  # Cut off
        
        # Check length
        if self.token_count < 50:
            indicators.append("⚠️SHORT")
        elif self.token_count > 600:
            indicators.append("⚠️LONG")
        
        # Check if starts with lowercase (likely mid-sentence)
        if text and text[0].islower():
            indicators.append("↪️CUT")
        
        return " ".join(indicators) if indicators else "🟢"
    
    def on_mount(self) -> None:
        """Apply dynamic styling on mount."""
        self.styles.border = ("solid", self.border_color)
        self.styles.background = self.bg_color
    
    def compose(self) -> ComposeResult:
        """Compose the chunk card content."""
        # Header with quality indicator
        quality = self._get_quality_indicators()
        header_text = f"█ Chunk {self.chunk_index + 1}  │  {len(self.chunk_text)} chars  │  ~{self.token_count} tok  │  {quality}"
        yield Static(header_text, classes="chunk-header")
        
        # Full content in scrollable container
        with VerticalScroll(classes="chunk-scroll"):
            yield Static(self.chunk_text, classes="chunk-content")
        
        # Metadata
        meta = f"📍 Position: {self.start_pos} → {self.end_pos}"
        yield Static(meta, classes="chunk-meta")


class ChunkList(Vertical):
    """Container for displaying multiple chunk cards."""
    
    DEFAULT_CSS = """
    ChunkList {
        height: 1fr;
        padding: 1;
        overflow-y: auto;
    }
    
    ChunkList .empty-state {
        color: $text-muted;
        text-align: center;
        padding: 4;
    }
    """
    
    def __init__(self, **kwargs):
        """Initialize the chunk list."""
        super().__init__(**kwargs)
        self._chunks = []
    
    def update_chunks(self, chunks: list) -> None:
        """Update the displayed chunks."""
        self._chunks = chunks
        self._rebuild_cards()
    
    def _rebuild_cards(self) -> None:
        """Rebuild all chunk cards."""
        self.remove_children()
        
        if not self._chunks:
            self.mount(Static("📭 No chunks yet. Load text to begin.", classes="empty-state"))
            return
        
        for i, (text, start, end) in enumerate(self._chunks):
            token_estimate = len(text) // 4
            card = ChunkCard(
                chunk_text=text,
                chunk_index=i,
                start_pos=start,
                end_pos=end,
                token_count=token_estimate,
                id=f"chunk-{i}"
            )
            self.mount(card)
