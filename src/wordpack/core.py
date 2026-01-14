"""
WordPack - Fill text glyphs with packed circles sampled from an image.

This module provides tools for creating circle-packed text art by:
1. Parsing font glyphs into polygon boundaries
2. Packing circles within those boundaries
3. Coloring circles based on an underlying image
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Generator, Tuple
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MplPolygon

from diskpack import PackingConfig, CirclePacker

# --- Type Aliases ---
Polygon = np.ndarray
GlyphPolygons = List[Polygon]


@dataclass
class FontConfig:
    """Configuration for font rendering and glyph parsing.
    
    Attributes:
        family: Font family name (e.g., 'serif', 'sans-serif', 'Arial')
        weight: Font weight (e.g., 'normal', 'bold')
        base_size: Base size for glyph rendering (affects resolution)
        letter_spacing: Multiplier for spacing between letters (1.0 = tight, 1.5 = loose)
        resolution: Bezier curve resolution (1=Blocky, 10=Natural, 50=Perfect)
    """
    family: str = "serif"
    weight: str = "bold"
    base_size: float = 100.0
    letter_spacing: float = 1.1
    resolution: int = 10


class GlyphParser:
    """Parses font characters into polygon boundaries for circle packing."""
    
    def __init__(self, config: FontConfig):
        """Initialize the parser with font configuration.
        
        Args:
            config: FontConfig instance specifying font properties
        """
        self.config = config
        self._font_props = FontProperties(family=config.family, weight=config.weight)

    def _get_bezier_points(
        self, 
        start: np.ndarray, 
        pts: np.ndarray, 
        num_points: int
    ) -> np.ndarray:
        """Calculate intermediate points along a Bezier curve.
        
        Args:
            start: Starting point of the curve
            pts: Control points (2 for quadratic, 3 for cubic)
            num_points: Number of intermediate points to generate
            
        Returns:
            Array of points along the Bezier curve
        """
        t = np.linspace(0, 1, num_points + 1)[1:]
        pts = pts.reshape(-1, 2)
        
        if len(pts) == 2:  # Quadratic (CURVE3)
            p1, p2 = pts
            return (
                (1-t)[:, None]**2 * start 
                + 2*(1-t)[:, None]*t[:, None] * p1 
                + t[:, None]**2 * p2
            )
        elif len(pts) == 3:  # Cubic (CURVE4)
            p1, p2, p3 = pts
            return (
                (1-t)[:, None]**3 * start 
                + 3*(1-t)[:, None]**2*t[:, None] * p1 
                + 3*(1-t)[:, None]*t[:, None]**2 * p2 
                + t[:, None]**3 * p3
            )
        return [pts[-1]]

    def parse_character(self, char: str) -> GlyphPolygons:
        """Parse a character into polygon boundaries.
        
        Converts a font glyph into isolated polygon loops suitable for
        circle packing. Handles complex glyphs with holes (e.g., 'O', 'A').
        
        Args:
            char: Single character to parse
            
        Returns:
            List of polygons (numpy arrays of shape (N, 2)) representing
            the glyph boundaries
        """
        path = TextPath((0, 0), char, size=self.config.base_size, prop=self._font_props)
        polygons: List[np.ndarray] = []
        current_pts: List[np.ndarray] = []
        last_pt: Optional[np.ndarray] = None

        for verts, code in path.iter_segments(simplify=False):
            if code == path.MOVETO:
                if len(current_pts) > 2:
                    polygons.append(np.array(current_pts))
                current_pts = [verts]
                last_pt = verts
            elif code == path.LINETO:
                if last_pt is None or not np.allclose(verts, last_pt):
                    current_pts.append(verts)
                    last_pt = verts
            elif code == path.CURVE3 or code == path.CURVE4:
                if last_pt is not None:
                    curve_segments = self._get_bezier_points(
                        last_pt, verts, self.config.resolution
                    )
                    current_pts.extend(curve_segments)
                else:
                    current_pts.append(verts[-2:])
                last_pt = verts[-2:]
            elif code == path.CLOSEPOLY:
                if len(current_pts) > 2:
                    polygons.append(np.array(current_pts))
                current_pts, last_pt = [], None

        if len(current_pts) > 2:
            polygons.append(np.array(current_pts))

        # Flip Y-axis to match image coordinates
        all_v = np.vstack(polygons)
        max_y = np.max(all_v[:, 1])
        return [p * [1, -1] + [0, max_y] for p in polygons]


class WordFiller:
    """Fill text with circles colored from an underlying image.
    
    This class orchestrates the process of:
    1. Loading a source image for colors
    2. Parsing text into glyph polygons
    3. Packing circles within the glyphs
    4. Rendering the result with matplotlib
    
    Example:
        >>> filler = WordFiller("background.jpg")
        >>> filler.run("HELLO", min_radius=2.0, padding=1.5)
    """
    
    def __init__(self, image_path: str):
        """Initialize with a source image.
        
        Args:
            image_path: Path to the image file used for circle colors
        """
        self.image = Image.open(image_path)
        self.W, self.H = self.image.size
        self.glyph_groups: List[GlyphPolygons] = []

    def run(self, text: str, **kwargs) -> None:
        """Run the circle packing visualization.
        
        Args:
            text: Text string to render
            **kwargs: Configuration options:
                - font_family (str): Font family name. Default: 'serif'
                - font_weight (str): Font weight. Default: 'bold'
                - resolution (int): Bezier curve resolution. Default: 10
                - letter_spacing (float): Letter spacing multiplier. Default: 1.1
                - padding (float): Circle padding. Default: 1.5
                - min_radius (float): Minimum circle radius. Default: 1.0
                - max_failed_attempts (int): Packing attempts before stopping. Default: 200
                - color_weights (dict): Optional color->probability mapping
                - margin_percent (float): Image margin percentage. Default: 0.1
                - fixed_radius (float): Optional fixed circle radius
                - batch_size (int): Circles per render update. Default: 40
                - show_outline (bool): Show glyph outlines. Default: True
        """
        # Build configurations
        f_cfg = FontConfig(
            family=kwargs.get('font_family', 'serif'),
            weight=kwargs.get('font_weight', 'bold'),
            resolution=kwargs.get('resolution', 10),
            letter_spacing=kwargs.get('letter_spacing', 1.1)
        )
        p_cfg = PackingConfig(
            padding=kwargs.get('padding', 1.5),
            min_radius=kwargs.get('min_radius', 1.0),
            max_failed_attempts=kwargs.get('max_failed_attempts', 200)
        )
        
        # Process color weights if provided
        color_weights = kwargs.get('color_weights')
        color_keys = None
        probs = None
        if color_weights:
            color_keys = list(color_weights.keys())
            probs = list(color_weights.values())
        
        # Parse glyphs
        parser = GlyphParser(f_cfg)
        raw = [parser.parse_character(c) for c in text]
        
        # Layout: position letters horizontally with spacing
        x_off = 0.0
        spaced: List[GlyphPolygons] = []
        for group in raw:
            pts = np.vstack(group)
            min_x, max_x = np.min(pts[:, 0]), np.max(pts[:, 0])
            offset = np.array([x_off - min_x, 0])
            spaced.append([p + offset for p in group])
            x_off += (max_x - min_x) * f_cfg.letter_spacing
            
        # Scale to fit image with margins
        all_pts = np.vstack([p for g in spaced for p in g])
        margin = kwargs.get('margin_percent', 0.1)
        scale = (self.W * (1 - 2 * margin)) / (np.max(all_pts[:, 0]) - np.min(all_pts[:, 0]))
        scaled = [[p * scale for p in g] for g in spaced]
        
        # Center in image
        all_s = np.vstack([p for g in scaled for p in g])
        offset = (
            np.array([self.W/2, self.H/2]) 
            - (np.min(all_s, axis=0) + np.max(all_s, axis=0)) / 2
        )
        self.glyph_groups = [[p + offset for p in g] for g in scaled]

        # Main packing loop
        fig, ax = plt.subplots(figsize=(10, 10))
        all_circles: List[plt.Circle] = []
        global_count = 0

        for idx, char in enumerate(text):
            packer = CirclePacker(self.glyph_groups[idx], p_cfg)
            gen = packer.generate(fixed_radius=kwargs.get('fixed_radius'))
            letter_count = 0
            
            while True:
                try:
                    x, y, r = next(gen)
                    
                    # Determine circle color
                    if color_weights and color_keys and probs:
                        color = np.random.choice(color_keys, p=probs)
                    else:
                        ix = int(np.clip(x, 0, self.W - 1))
                        iy = int(np.clip(y, 0, self.H - 1))
                        color = np.array(self.image.getpixel((ix, iy))) / 255.0
                    
                    all_circles.append(plt.Circle((x, y), r, facecolor=color))
                    letter_count += 1
                    global_count += 1
                    
                    # Periodic rendering
                    batch_size = kwargs.get('batch_size', 40)
                    if letter_count % batch_size == 0:
                        self._render(
                            ax, fig, all_circles, char, 
                            letter_count, global_count, 
                            kwargs.get('show_outline', True)
                        )
                except StopIteration:
                    self._render(
                        ax, fig, all_circles, char, 
                        letter_count, global_count, 
                        kwargs.get('show_outline', True)
                    )
                    break

    def _render(
        self, 
        ax: plt.Axes, 
        fig: plt.Figure, 
        circles: List[plt.Circle], 
        char: str, 
        letter_count: int, 
        global_count: int, 
        outline: bool
    ) -> None:
        """Render the current state of the visualization.
        
        Args:
            ax: Matplotlib axes
            fig: Matplotlib figure
            circles: List of Circle patches to render
            char: Current character being processed
            letter_count: Number of circles in current letter
            global_count: Total number of circles
            outline: Whether to show glyph outlines
        """
        try:
            from IPython.display import clear_output, display
            clear_output(wait=True)
            print(f"Current Letter: '{char}' ({letter_count} circles) | Total Circles: {global_count}")
        except ImportError:
            pass
            
        ax.clear()
        ax.imshow(self.image)
        
        if outline:
            for g in self.glyph_groups:
                for p in g:
                    ax.add_patch(MplPolygon(
                        p, closed=True, fill=None, 
                        edgecolor='white', lw=1, ls='--', alpha=0.4
                    ))
        
        ax.add_collection(PatchCollection(
            circles, match_original=True, edgecolor='black', lw=0.1
        ))
        ax.axis('off')
        
        try:
            from IPython.display import display
            display(fig)
        except ImportError:
            plt.pause(0.01)
