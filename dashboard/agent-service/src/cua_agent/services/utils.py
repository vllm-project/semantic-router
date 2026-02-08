"""Utility functions for services."""

from io import BytesIO

from PIL import Image


def compress_image_to_max_size(
    image: Image.Image, max_size_kb: int = 800
) -> Image.Image:
    """
    Compress an image to ensure it doesn't exceed max_size_kb.
    Always returns a PIL Image in PNG format.
    Resizes the image if necessary to meet the size requirement.

    Args:
        image: PIL Image to compress
        max_size_kb: Maximum size in KB (default: 800KB)

    Returns:
        PIL Image in PNG format that meets the size requirement
    """
    max_size_bytes = max_size_kb * 1024
    current_image = image.copy()

    # Try PNG compression first with different compression levels
    for compress_level in [3, 6, 9]:
        buffered = BytesIO()
        current_image.save(buffered, format="PNG", compress_level=compress_level)
        png_bytes = buffered.getvalue()
        buffered.close()
        png_size = len(png_bytes)
        if png_size <= max_size_bytes:
            return current_image

    # PNG is too large even with maximum compression, need to resize
    scale_factor = 1.0
    min_scale = 0.1
    max_scale = 1.0
    iterations = 0
    max_iterations = 10
    best_image = None

    while iterations < max_iterations:
        # Resize image
        new_size = (
            int(image.size[0] * scale_factor),
            int(image.size[1] * scale_factor),
        )
        resized_image = image.resize(new_size, Image.Resampling.LANCZOS)

        # Check PNG size with best compression
        buffered = BytesIO()
        resized_image.save(buffered, format="PNG", compress_level=9)
        png_bytes = buffered.getvalue()
        buffered.close()
        png_size = len(png_bytes)

        if png_size <= max_size_bytes:
            best_image = resized_image
            if scale_factor >= max_scale - 0.01:
                return resized_image
            min_scale = scale_factor
            scale_factor = (scale_factor + max_scale) / 2
        else:
            max_scale = scale_factor
            scale_factor = (min_scale + scale_factor) / 2

        iterations += 1

    if best_image is not None:
        return best_image

    # Final fallback: resize to minimum scale
    final_size = (
        int(image.size[0] * min_scale),
        int(image.size[1] * min_scale),
    )
    final_image = image.resize(final_size, Image.Resampling.LANCZOS)
    return final_image
