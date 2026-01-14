"""
WordPack - Create circle-packed text art from images.

This package provides tools for filling text glyphs with circles
whose colors are sampled from an underlying image.

Example:
    >>> from wordpack import WordFiller
    >>> filler = WordFiller("my_image.jpg")
    >>> filler.run("HELLO", min_radius=2.0, padding=1.5)

Classes:
    WordFiller: Main class for creating circle-packed text visualizations
    FontConfig: Configuration for font rendering
    GlyphParser: Parser for converting fonts to polygon boundaries
    PackingConfig: Configuration for circle packing algorithm (from diskpack)
    CirclePacker: Circle packing implementation (from diskpack)
"""

from .core import WordFiller, FontConfig, GlyphParser

from diskpack import PackingConfig, CirclePacker

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

__all__ = [
    "WordFiller",
    "FontConfig", 
    "GlyphParser",
    "PackingConfig",
    "CirclePacker",
]
