import sys
import pathlib

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib.patches import Ellipse, Circle, Polygon, PathPatch
from matplotlib.path import Path

PATH = pathlib.Path("/home/pku/Astro/FEASTS_SED/data")

class GalaxyLabeler:
    def __init__(self, images, bands, save_path='pixbin_morph.npy', stretch_scale=5.0e4):
        self.images = images
        self.bands = bands
        self.save_path = save_path
        self.stretch_scale = stretch_scale
        self.image_size = images[0].shape[0]
        
        # self.fig, self.axes = plt.subplots(1, len(bands), figsize=(18, 7))
        self.fig = plt.figure(figsize=(10, 10))
        _girdspec = plt.GridSpec(3, 2, wspace=0.05, hspace=0.05)
        self.axes = [self.fig.add_subplot(_girdspec[0, 0]),
                     self.fig.add_subplot(_girdspec[1:,:]),
                     self.fig.add_subplot(_girdspec[0, 1])]
        self.current_band = 1  # Middle image (r-band)
        # self.mask = np.zeros(images[0].shape, dtype=np.uint8)
        self.mask = np.full(images[0].shape, -32768, dtype=np.int16)
        self.current_label = 1
        self.drawing = False
        self.current_shape = 'polygon'
        self.current_points = []
        self.patches = {band: [] for band in bands}  # Permanent patches
        self.temp_shapes = []  # Temporary preview shapes
        self.start_pos = None

        # Initialize plots with arcsinh stretching
        for ax, band, img in zip(self.axes, bands, images):
            stretched_img = self.arcsinh_stretch(img, self.stretch_scale)
            ax.imshow(stretched_img, cmap='gray', interpolation='nearest')
            ax.set_title(f'{band}-band')
            ax.set_xticks([])
            ax.set_yticks([])

        self.axes[1].text(0.5, -0.1, '1=polygon, 2=circle, 3=ellipse, m=save mask',
                          transform=self.axes[1].transAxes, 
                          ha='center', fontsize=12, color='blue')
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

    @staticmethod
    def arcsinh_stretch(image, scale):
        """Apply arcsinh stretching to enhance faint features"""
        stretched = np.arcsinh(image * scale)
        return stretched / np.arcsinh(scale)

    def update_overlays(self):
        """Update mask overlays on all band images"""
        for ax, band in zip(self.axes, self.bands):
            # Clear existing patches
            for p in self.patches[band]:
                p.remove()
            self.patches[band] = []
            
            # Create new PathPatches from contours
            unique_labels = np.unique(self.mask)
            unique_labels = unique_labels[unique_labels != 0]
            levels = [ll - 0.5 for ll in unique_labels]
            if len(levels) > 0:
                contours = self.axes[0].contour(self.mask, levels=levels, colors='r', alpha=0)
                for collection in contours.collections:
                    for path in collection.get_paths():
                        patch = PathPatch(path, facecolor='none', edgecolor='r')
                        self.patches[band].append(ax.add_patch(patch))
            
        self.fig.canvas.draw_idle()

    def clear_temp_shapes(self):
        """Remove all temporary drawing shapes"""
        for shape in self.temp_shapes:
            shape.remove()
        self.temp_shapes = []

    def draw_temp_shape(self, event=None):
        """Draw temporary shapes during creation"""
        self.clear_temp_shapes()

        if self.current_shape == 'polygon' and len(self.current_points) >= 1:
            temp_points = self.current_points.copy()
            if event and self.drawing:
                temp_points.append((event.xdata, event.ydata))
            
            if len(temp_points) > 1:
                for ax in self.axes:
                    new_poly = Polygon(temp_points, closed=False, ec='r', fill=False, lw=1)
                    self.temp_shapes.append(ax.add_patch(new_poly))
        
        elif self.current_shape == 'circle' and self.start_pos and event:
            radius = np.hypot(event.xdata - self.start_pos[0],
                              event.ydata - self.start_pos[1])
            for ax in self.axes:
                circ = Circle(self.start_pos, radius, ec='r', fill=False, lw=1)
                self.temp_shapes.append(ax.add_patch(circ))
        
        elif self.current_shape == 'ellipse' and self.start_pos and event:
            width  = abs(event.xdata - self.start_pos[0]) * 2
            height = abs(event.ydata - self.start_pos[1]) * 2
            for ax in self.axes:
                ellipse = Ellipse(self.start_pos, width, height, ec='r', fill=False, lw=1)
                self.temp_shapes.append(ax.add_patch(ellipse))
        
        self.fig.canvas.draw_idle()

    def _fill_mask_gaps(self):
        """Fill small gaps in the mask using binary dilation and nearest neighbor labeling"""
        from scipy.ndimage import binary_dilation, label
        from scipy.stats import mode

        # dilation
        for i in range(1, np.max(self.mask) + 1):
            dilate_mask = binary_dilation(self.mask == i)
            self.mask[dilate_mask] = i
        print("Mask gaps filled with dilation")

        # use nearest neighbor labeling to fill small gaps
        mask_label, _ = label(self.mask < 1)
        for i in range(1, np.max(mask_label) + 1):
            if np.sum(mask_label == i) > self.image_size**2 * 0.01:
                mask_label[mask_label == i] = 0
        for i, j in zip(*np.where(mask_label > 0)):
            self.mask[i][j], _ = mode(self.mask[max(0, i-1):min(self.image_size, i+2),
                                        max(0, j-1):min(self.image_size, j+2)].flatten())
        print(f"Remaining {np.sum(mask_label)} pixels filled with nearest labels")

        self.mask[self.mask < 0] = 0 # Ensure no negative values in mask

    def on_click(self, event):
        if event.inaxes != self.axes[self.current_band]:
            return
            
        if self.current_shape == 'polygon':
            if not self.drawing:
                # Start new polygon
                self.current_points = [(event.xdata, event.ydata)]
                self.drawing = True
                # Add first vertex marker to permanent patches
                for ax, band in zip(self.axes, self.bands):
                    circ = Circle((event.xdata, event.ydata), radius=1, ec='r', fill=False)
                    self.patches[band].append(ax.add_patch(circ))
            else:
                # Add subsequent vertex
                self.current_points.append((event.xdata, event.ydata))

                # Add vertex marker to permanent patches
                for ax, band in zip(self.axes, self.bands):
                    circ = Circle((event.xdata, event.ydata), radius=1, ec='r', fill=False)
                    self.patches[band].append(ax.add_patch(circ))
            
            self.draw_temp_shape()  # Update preview line
            self.fig.canvas.draw_idle()

        elif self.current_shape in ['circle', 'ellipse']:
            self.start_pos = (event.xdata, event.ydata)
            self.drawing = True

    def on_motion(self, event):
        if not self.drawing or event.inaxes != self.axes[self.current_band]:
            return
        self.draw_temp_shape(event)

    def on_release(self, event):
        if not self.drawing or event.inaxes != self.axes[self.current_band]:
            return
            
        if self.current_shape == 'circle':
            radius = np.hypot(event.xdata - self.start_pos[0], 
                            event.ydata - self.start_pos[1])
            y, x = np.ogrid[-self.start_pos[1]:self.image_size-self.start_pos[1], 
                           -self.start_pos[0]:self.image_size-self.start_pos[0]]
            mask = x*x + y*y <= radius*radius
            self.mask[mask] = self.current_label
            self.current_label += 1
            self.update_overlays()
        elif self.current_shape == 'ellipse':
            width = abs(event.xdata - self.start_pos[0]) * 2
            height = abs(event.ydata - self.start_pos[1]) * 2
            y, x = np.ogrid[-self.start_pos[1]:self.image_size-self.start_pos[1], 
                           -self.start_pos[0]:self.image_size-self.start_pos[0]]
            mask = (x**2)/((width/2)**2) + (y**2)/((height/2)**2) <= 1
            self.mask[mask] = self.current_label
            self.current_label += 1
            self.update_overlays()

        if not self.current_shape == 'polygon':
            self.drawing = False
        self.start_pos = None
        self.clear_temp_shapes()

    def on_key(self, event):
        if event.key == 'escape':
            self.current_shape = None
        elif event.key == '1':
            self.current_shape = 'polygon'
            print("Polygon tool: Click vertices, press Enter to finish")
        elif event.key == '2':
            self.current_shape = 'circle'
            print("Circle tool: Click-and-drag")
        elif event.key == '3':
            self.current_shape = 'ellipse'
            print("Ellipse tool: Click-and-drag")
        elif event.key == 'enter' and self.current_shape == 'polygon':
            if len(self.current_points) > 2:
                # Create closed polygon path
                poly = Path(self.current_points + [self.current_points[0]], closed=True)  # Explicitly close the polygon
                
                # Generate grid covering all pixels
                x = np.arange(0.5, self.image_size + 0.5)
                y = np.arange(0.5, self.image_size + 0.5)
                xv, yv = np.meshgrid(x, y)
                points = np.vstack((xv.flatten(), yv.flatten())).T
                
                # Create mask (ensure proper dtype)
                mask = poly.contains_points(points).reshape(self.image_size, self.image_size)
                self.mask[mask] = self.current_label
                self.current_label += 1
                self.update_overlays()
                print(f"Polygon completed with label {self.current_label-1}")
                
                # Draw the final closed polygon outline
                for ax, band in zip(self.axes, self.bands):
                    poly_patch = Polygon(self.current_points, closed=True, ec='r', fill=False)
                    self.patches[band].append(ax.add_patch(poly_patch))
            
            self.drawing = False
            self.current_points = []
            self.clear_temp_shapes()
            self.fig.canvas.draw_idle()  # Force immediate update
        elif event.key == 'c': # clear selected label
            if self.current_label:
                self.mask[self.mask == self.current_label - 1] = 0
                self.current_label -= 1
                print(f"Cleared label {self.current_label}")
                self.update_overlays()
        elif event.key == 'm':
            self._fill_mask_gaps()
            np.save(self.save_path, self.mask)
            print(f"Mask saved to {self.save_path}")


if __name__ == "__main__":
    bands = ['GALEX_NUV', 'SDSS_r', 'SDSS_i']
    gname = sys.argv[1]
    images = []
    
    for fname in bands:
        hdul = fits.open(PATH / gname / f"stamp_psfmatch_crop_skybgsub_{gname}_{fname}.fits")
        # hdul = fits.open(f"./test_data/stamp_psfmatch_crop_skybgsub_{gname}_{fname}.fits")
        images.append(hdul[0].data)
    
    labeler = GalaxyLabeler(images, bands, save_path=PATH / gname / "pixbin_morph.npy")
    # labeler = GalaxyLabeler(images, bands, save_path=f"./test_data/pixbin_morph_{gname}.npy")
    plt.show()