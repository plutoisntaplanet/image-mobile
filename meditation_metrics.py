"""Small, deterministic image measurements for Meditation; image bytes stay in memory."""
from io import BytesIO

from PIL import Image, ImageChops, ImageFilter, ImageStat


def measure_image(image_bytes):
    """Return size and high-frequency detail. Compare detail only for the same scene."""
    with Image.open(BytesIO(image_bytes)) as source:
        width, height = source.size
        gray = source.convert('L')
    gray.thumbnail((512, 512), Image.Resampling.LANCZOS)
    high = ImageChops.difference(gray, gray.filter(ImageFilter.GaussianBlur(2)))
    return {'width': width, 'height': height,
            'detail_energy': round(ImageStat.Stat(high).rms[0] / 255, 6)}
