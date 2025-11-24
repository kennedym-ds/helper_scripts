# Image Comparison Tool

A Python-based image comparison tool with a simple GUI built using Panel.

## Features

- **Side-by-side image comparison**: Upload and view two images simultaneously
- **Similarity metrics**: Calculate Mean Squared Error (MSE) and Structural Similarity Index (SSIM)
- **Difference visualization**: See a visual representation of the differences between images
- **Simple GUI**: Clean and intuitive interface built with Panel

## Requirements

- Python 3.8+
- panel >= 1.0.0
- pillow >= 8.0.0
- scikit-image >= 0.18.0
- matplotlib >= 3.5.0
- numpy >= 1.21.0

## Installation

Install the required dependencies:

```bash
pip install panel pillow scikit-image matplotlib numpy
```

Or install from the requirements file:

```bash
pip install -r learning_series/requirements.txt
```

## Usage

### Running the application

To start the image comparison tool:

```bash
python image_compare.py
```

This will launch a web server and open your default browser to the application interface (typically at http://localhost:5006).

### Using the GUI

1. **Upload Images**: Click on "Upload First Image" and "Upload Second Image" to select your images
2. **Compare**: Once both images are loaded, click the "Compare Images" button
3. **View Results**:
   - The two images will be displayed side-by-side
   - A difference visualization will show where the images differ
   - Metrics will display:
     - Mean Squared Error (MSE) - lower is better (0 = identical)
     - Structural Similarity Index (SSIM) - higher is better (1 = identical)
     - Image dimensions and similarity percentage

### Programmatic Usage

You can also use the ImageCompare class in your own code:

```python
from image_compare import ImageCompare

# Create an instance
app = ImageCompare()

# Serve the application (opens in browser)
app.serve(port=5006)

# Or create the layout without serving
layout = app.create_layout()
```

## Metrics Explained

### Mean Squared Error (MSE)
- Measures the average squared difference between pixel values
- Range: 0 to ∞
- Lower values indicate more similarity
- 0 means the images are identical

### Structural Similarity Index (SSIM)
- Measures perceived quality and structural similarity
- Range: -1 to 1
- Higher values indicate more similarity
- 1 means the images are identical
- Better reflects human perception than MSE

## Image Handling

- The tool automatically handles different image formats (PNG, JPG, JPEG, BMP, GIF)
- Images with different dimensions will be automatically resized to match for comparison
- RGBA images are converted to RGB for comparison
- Grayscale images are converted to RGB

## Example Use Cases

- **Quality Assurance**: Compare original and processed images to verify processing quality
- **Version Control**: Compare different versions of an image to see what changed
- **Design Review**: Compare design mockups with implemented results
- **Data Validation**: Verify image transformations and augmentations in ML pipelines
- **Testing**: Compare reference images with generated outputs in automated tests

## Troubleshooting

**Port already in use**: If port 5006 is already in use, you can specify a different port:
```python
app.serve(port=5007)
```

**Images not displaying**: Make sure your images are in a supported format (PNG, JPG, JPEG, BMP, GIF)

**Different sized images**: The tool will automatically resize the second image to match the first for comparison

## License

This tool is part of the helper_scripts repository.
