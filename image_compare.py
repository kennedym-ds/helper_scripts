"""
Image Comparison Tool with Panel GUI

This module provides a simple GUI for comparing two images side-by-side
with various comparison metrics and visualizations.
"""

import panel as pn
import numpy as np
from PIL import Image
import io
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Initialize Panel extension
pn.extension(sizing_mode='stretch_width')


class ImageCompare:
    """
    A Panel-based GUI application for comparing two images.
    
    Features:
    - Load two images via file upload
    - Display images side-by-side
    - Calculate and display similarity metrics (MSE, SSIM)
    - Show difference visualization
    """
    
    def __init__(self):
        self.image1 = None
        self.image2 = None
        self.image1_array = None
        self.image2_array = None
        
        # Create widgets
        self.file_input1 = pn.widgets.FileInput(
            name='Upload First Image',
            accept='.png,.jpg,.jpeg,.bmp,.gif',
            multiple=False
        )
        self.file_input2 = pn.widgets.FileInput(
            name='Upload Second Image',
            accept='.png,.jpg,.jpeg,.bmp,.gif',
            multiple=False
        )
        
        self.compare_button = pn.widgets.Button(
            name='Compare Images',
            button_type='primary',
            disabled=True
        )
        
        # Create display panes
        self.image1_pane = pn.pane.PNG(
            object=None,
            width=400,
            height=400,
            name='Image 1'
        )
        self.image2_pane = pn.pane.PNG(
            object=None,
            width=400,
            height=400,
            name='Image 2'
        )
        self.diff_pane = pn.pane.PNG(
            object=None,
            width=400,
            height=400,
            name='Difference'
        )
        
        self.metrics_pane = pn.pane.Markdown(
            "Upload two images to compare them.",
            max_width=400
        )
        
        # Set up callbacks
        self.file_input1.param.watch(self._on_image1_upload, 'value')
        self.file_input2.param.watch(self._on_image2_upload, 'value')
        self.compare_button.on_click(self._compare_images)
        
    def _on_image1_upload(self, event):
        """Handle first image upload"""
        if event.new:
            self.image1 = Image.open(io.BytesIO(event.new))
            self.image1_array = np.array(self.image1)
            # Convert to RGB if necessary
            if len(self.image1_array.shape) == 2:
                self.image1_array = np.stack([self.image1_array] * 3, axis=-1)
            elif self.image1_array.shape[2] == 4:
                self.image1_array = self.image1_array[:, :, :3]
            
            self.image1_pane.object = self.image1
            self._update_button_state()
    
    def _on_image2_upload(self, event):
        """Handle second image upload"""
        if event.new:
            self.image2 = Image.open(io.BytesIO(event.new))
            self.image2_array = np.array(self.image2)
            # Convert to RGB if necessary
            if len(self.image2_array.shape) == 2:
                self.image2_array = np.stack([self.image2_array] * 3, axis=-1)
            elif self.image2_array.shape[2] == 4:
                self.image2_array = self.image2_array[:, :, :3]
            
            self.image2_pane.object = self.image2
            self._update_button_state()
    
    def _update_button_state(self):
        """Enable compare button when both images are loaded"""
        self.compare_button.disabled = not (self.image1 and self.image2)
    
    def _compare_images(self, event):
        """Compare the two loaded images"""
        if not (self.image1 and self.image2):
            self.metrics_pane.object = "**Error:** Please upload both images."
            return
        
        # Resize images to match if they have different dimensions
        if self.image1_array.shape != self.image2_array.shape:
            # Resize image2 to match image1
            img2_resized = Image.fromarray(self.image2_array)
            img2_resized = img2_resized.resize(
                (self.image1_array.shape[1], self.image1_array.shape[0]),
                Image.Resampling.LANCZOS
            )
            image2_array = np.array(img2_resized)
        else:
            image2_array = self.image2_array
        
        # Calculate metrics
        mse_value = mean_squared_error(self.image1_array, image2_array)
        
        # Convert to grayscale for SSIM
        img1_gray = np.mean(self.image1_array, axis=2).astype(np.uint8)
        img2_gray = np.mean(image2_array, axis=2).astype(np.uint8)
        ssim_value = ssim(img1_gray, img2_gray)
        
        # Calculate difference image
        diff = np.abs(self.image1_array.astype(float) - image2_array.astype(float))
        diff_normalized = (diff / diff.max() * 255).astype(np.uint8) if diff.max() > 0 else diff.astype(np.uint8)
        
        # Create difference visualization
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(diff_normalized)
        ax.set_title('Absolute Difference')
        ax.axis('off')
        
        # Convert matplotlib figure to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        plt.close(fig)
        
        diff_image = Image.open(buf)
        self.diff_pane.object = diff_image
        
        # Update metrics display
        metrics_text = f"""
## Comparison Metrics

**Mean Squared Error (MSE):** {mse_value:.2f}
- Lower is better (0 = identical)

**Structural Similarity Index (SSIM):** {ssim_value:.4f}
- Range: [-1, 1], Higher is better (1 = identical)

**Image 1 Size:** {self.image1_array.shape[1]} x {self.image1_array.shape[0]}

**Image 2 Size:** {self.image2_array.shape[1]} x {self.image2_array.shape[0]}

**Similarity:** {ssim_value * 100:.2f}%
"""
        self.metrics_pane.object = metrics_text
    
    def create_layout(self):
        """Create the Panel layout"""
        title = pn.pane.Markdown("# Image Comparison Tool", styles={'color': '#333'})
        description = pn.pane.Markdown(
            "Upload two images to compare them side-by-side. "
            "The tool will calculate similarity metrics and show the differences."
        )
        
        # Upload section
        upload_section = pn.Column(
            pn.pane.Markdown("## Upload Images"),
            pn.Row(self.file_input1, self.file_input2),
            self.compare_button,
            max_width=900
        )
        
        # Display section
        images_row = pn.Row(
            pn.Column(pn.pane.Markdown("### Image 1"), self.image1_pane),
            pn.Column(pn.pane.Markdown("### Image 2"), self.image2_pane),
            pn.Column(pn.pane.Markdown("### Difference"), self.diff_pane),
            align='start'
        )
        
        # Metrics section
        metrics_section = pn.Column(
            pn.pane.Markdown("## Metrics"),
            self.metrics_pane
        )
        
        # Complete layout
        layout = pn.Column(
            title,
            description,
            upload_section,
            images_row,
            metrics_section,
            sizing_mode='stretch_width',
            max_width=900
        )
        
        return layout
    
    def serve(self, port=5006, show=True):
        """Serve the application"""
        layout = self.create_layout()
        return pn.serve(layout, port=port, show=show, title='Image Compare')


def main():
    """Main entry point for the application"""
    app = ImageCompare()
    app.serve()


if __name__ == '__main__':
    main()
