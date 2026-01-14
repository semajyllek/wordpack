# WordPack

Create stunning circle-packed text art by filling typography with circles colored from an underlying image. Built on top of [diskpack](https://pypi.org/project/diskpack/).

![Example Output](examples/example_output.png)

## Features

- **Font Flexibility**: Works with any system font (serif, sans-serif, custom fonts)
- **Bezier Curve Support**: Smooth glyph parsing with configurable resolution
- **Image-Based Coloring**: Sample colors from any source image
- **Custom Color Weights**: Override image colors with weighted random selection
- **Live Preview**: Watch the packing process in real-time (Jupyter notebooks)
- **Configurable Packing**: Control circle sizes, padding, and density

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/wordpack.git
cd wordpack

# Install in development mode
pip install -e .

# Or install with dev dependencies
pip install -e ".[dev]"

# For Jupyter notebook support
pip install -e ".[jupyter]"
```

## Quick Start

```python
from wordpack import WordFiller

# Create a filler with your source image
filler = WordFiller("path/to/your/image.jpg")

# Generate circle-packed text
filler.run(
    "HELLO",
    min_radius=1.5,
    padding=1.0,
    font_family="Arial",
    font_weight="bold"
)
```

## Configuration Options

### Font Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `font_family` | str | "serif" | Font family name |
| `font_weight` | str | "bold" | Font weight |
| `resolution` | int | 10 | Bezier curve resolution (1=blocky, 50=perfect) |
| `letter_spacing` | float | 1.1 | Spacing multiplier between letters |

### Packing Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_radius` | float | 1.0 | Minimum circle radius |
| `padding` | float | 1.5 | Space between circles |
| `max_failed_attempts` | int | 200 | Failed attempts before stopping |
| `fixed_radius` | float | None | Use fixed radius for all circles |
| `sample_batch_size` | int | - | Batch size for sampling (diskpack) |
| `grid_resolution_divisor` | int | - | Grid resolution control (diskpack) |
| `mega_circle_threshold` | float | - | Threshold for large circles (diskpack) |

### Display Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `margin_percent` | float | 0.1 | Image margin (0-0.5) |
| `batch_size` | int | 40 | Circles per render update |
| `show_outline` | bool | True | Show glyph outlines |

### Color Options

```python
# Use image colors (default)
filler.run("TEXT")

# Use custom weighted colors
filler.run(
    "TEXT",
    color_weights={
        "#FF0000": 0.5,  # 50% red
        "#00FF00": 0.3,  # 30% green
        "#0000FF": 0.2,  # 20% blue
    }
)
```

## Advanced Usage

### Custom Font Configuration

```python
from wordpack import FontConfig, GlyphParser

config = FontConfig(
    family="Georgia",
    weight="bold",
    base_size=150.0,
    letter_spacing=1.2,
    resolution=20
)

parser = GlyphParser(config)
polygons = parser.parse_character("A")
```

### Using diskpack Directly

This package uses [diskpack](https://pypi.org/project/diskpack/) for circle packing. You can also use it directly for more control:

```python
from diskpack import PackingConfig, CirclePacker

config = PackingConfig(
    padding=0.7,
    min_radius=0.25,
    max_failed_attempts=700,
    sample_batch_size=100,
    grid_resolution_divisor=40,
    mega_circle_threshold=0.3,
)

packer = CirclePacker(polygons, config)
for x, y, r in packer.generate():
    # Process each circle
    pass
```

## Examples

See the `examples/` directory for Jupyter notebooks demonstrating various use cases:

- `basic_usage.ipynb` - Simple getting started example
- `custom_colors.ipynb` - Using color weights
- `font_exploration.ipynb` - Trying different fonts

## Dependencies

- Python >= 3.8
- numpy >= 1.20.0
- Pillow >= 8.0.0
- matplotlib >= 3.4.0
- [diskpack](https://pypi.org/project/diskpack/) >= 0.1.0

## How It Works

1. **Glyph Parsing**: Text characters are converted to polygon boundaries using matplotlib's `TextPath`. Bezier curves are sampled at configurable resolution.

2. **Layout**: Glyphs are positioned horizontally with configurable spacing, then scaled and centered on the source image.

3. **Circle Packing**: Circles are packed within glyph boundaries using rejection sampling. Each circle's color is sampled from the underlying image pixel.

4. **Rendering**: Matplotlib renders the packed circles with optional glyph outlines for debugging.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by generative typography and circle packing algorithms
- Built with matplotlib's excellent font and path handling
