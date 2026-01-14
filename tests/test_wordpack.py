"""Tests for the wordpack package."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from wordpack import FontConfig, GlyphParser


class TestFontConfig:
    """Tests for FontConfig dataclass."""
    
    def test_default_values(self):
        config = FontConfig()
        assert config.family == "serif"
        assert config.weight == "bold"
        assert config.base_size == 100.0
        assert config.letter_spacing == 1.1
        assert config.resolution == 10
    
    def test_custom_values(self):
        config = FontConfig(
            family="Arial",
            weight="normal",
            base_size=50.0,
            letter_spacing=1.5,
            resolution=20
        )
        assert config.family == "Arial"
        assert config.weight == "normal"
        assert config.base_size == 50.0
        assert config.letter_spacing == 1.5
        assert config.resolution == 20


class TestGlyphParser:
    """Tests for GlyphParser class."""
    
    def test_parse_simple_character(self):
        config = FontConfig(resolution=5)
        parser = GlyphParser(config)
        
        # Parse a simple character
        polygons = parser.parse_character("I")
        
        # Should return list of polygons
        assert isinstance(polygons, list)
        assert len(polygons) > 0
        
        # Each polygon should be a numpy array
        for poly in polygons:
            assert isinstance(poly, np.ndarray)
            assert poly.ndim == 2
            assert poly.shape[1] == 2  # x, y coordinates
    
    def test_parse_character_with_hole(self):
        config = FontConfig(resolution=5)
        parser = GlyphParser(config)
        
        # Characters like 'O' have holes, should produce multiple polygons
        polygons = parser.parse_character("O")
        
        assert len(polygons) >= 2  # Outer and inner boundary
    
    def test_bezier_points_quadratic(self):
        config = FontConfig()
        parser = GlyphParser(config)
        
        start = np.array([0.0, 0.0])
        pts = np.array([1.0, 1.0, 2.0, 0.0])  # control point, end point
        
        result = parser._get_bezier_points(start, pts, num_points=3)
        
        assert len(result) == 3
        # End point should be close to (2, 0)
        assert np.allclose(result[-1], [2.0, 0.0], atol=0.01)


class TestWordFiller:
    """Tests for WordFiller class (with mocked image)."""
    
    @patch('wordpack.core.Image')
    def test_initialization(self, mock_image_module):
        from wordpack import WordFiller
        
        # Mock the image
        mock_img = MagicMock()
        mock_img.size = (800, 600)
        mock_image_module.open.return_value = mock_img
        
        filler = WordFiller("fake_path.jpg")
        
        assert filler.W == 800
        assert filler.H == 600
        mock_image_module.open.assert_called_once_with("fake_path.jpg")


# Integration tests for diskpack (only run if diskpack is installed)
class TestDiskpackIntegration:
    """Integration tests with diskpack package."""
    
    def test_packing_config_import(self):
        from diskpack import PackingConfig
        
        config = PackingConfig(
            padding=0.7,
            min_radius=0.25,
            max_failed_attempts=700,
        )
        assert config.padding == 0.7
        assert config.min_radius == 0.25
    
    def test_circle_packer_import(self):
        from diskpack import CirclePacker
        assert CirclePacker is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

