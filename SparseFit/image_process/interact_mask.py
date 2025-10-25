import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')  # Set backend before pyplot import
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import ImageNormalize, AsinhStretch
from matplotlib.patches import Circle 

class CircleMaskCreator:
    def __init__(self, fits_file, output_path):
        self.fits_file = fits_file
        self.output_path = output_path
        self.original_data = fits.getdata(fits_file)
        self.mode = "add"
        self.vmin = 5.
        self.vmax = 99.8
        self.stretch_factor = 0.1
        
        # Create figure with adaptive size
        aspect_ratio = self.original_data.shape[1]/self.original_data.shape[0]
        figsize = (10 * aspect_ratio, 10)
        
        self.fig  = plt.figure(figsize=figsize)
        self.canvas = self.fig.canvas
        self.ax = self.fig.add_subplot(111)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # Try to load existing mask
        self.circles = []
        try:
            existing_mask = np.load(output_path)
            self.load_existing_mask(existing_mask)
            print(f"Loaded existing mask from {output_path}")
        except (FileNotFoundError, OSError):
            print("No existing mask found - starting fresh")
        
        # Smart visualization parameters
        self.update_display()

        # Initialize mask elements
        self.current_circle = None
        self.dragging = False
        self.press_pos = None

        # Connect events using the canvas
        self.cid_press = self.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_motion = self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cid_release = self.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_key = self.canvas.mpl_connect('key_press_event', self.on_key_press)
        

    def update_display(self):
        """Update the image display with current stretch settings"""
        self.norm = ImageNormalize(
            self.original_data, 
            vmin=np.nanpercentile(self.original_data, self.vmin),
            vmax=np.nanpercentile(self.original_data, self.vmax),
            stretch=AsinhStretch(a=self.stretch_factor),
            clip=False
        )
        
        # Remove only the image (not other elements)
        # for img in self.ax.images:
        #     img.remove()
        self.ax.clear()
        
        # Redraw image
        self.ax.imshow(self.original_data, origin='lower', 
                       cmap='gray', norm=self.norm)
        self.ax.text(0.01, 0.01, self.fits_file, 
            transform=self.ax.transAxes, fontsize=8, bbox=dict(fc='w', alpha=.7))
        self.ax.text(0.99, 0.01, "1:Add | 2:Remove | m:Save | C:ClearAll", 
            transform=self.ax.transAxes, ha='right', va='bottom', fontsize=8, bbox=dict(fc='w', alpha=.7))
        self.ax.set_axis_off()
        for circ in self.circles:
            self.ax.add_patch(circ)
        
        # Redraw all circles (they persist through updates)
        self.canvas.draw_idle()
        
    def on_press(self, event):
        if event.inaxes != self.ax or event.button != 1:
            return

        if self.mode == "add":
            self.press_pos = (event.xdata, event.ydata)
            self.current_circle = Circle(  # Using Circle instead of Ellipse
                (event.xdata, event.ydata), 
                radius=0,  # Using radius instead of width/height
                edgecolor='r', facecolor='none', lw=0.5
            )
            self.ax.add_patch(self.current_circle)
            self.dragging = True
            
        elif self.mode == "remove":
            min_dist = np.inf
            to_remove = None
            for c in self.circles:
                dx = c.center[0] - event.xdata
                dy = c.center[1] - event.ydata
                dist = np.hypot(dx, dy)/c.radius  # Simplified distance check
                if dist < min_dist and dist < 0.5:
                    min_dist = dist
                    to_remove = c
            
            if to_remove:
                to_remove.remove()
                self.circles.remove(to_remove)
                self.fig.canvas.draw()

    def on_motion(self, event):
        if not self.dragging or event.inaxes != self.ax:
            return
        
        if self.current_circle:
            x0, y0 = self.press_pos
            radius = np.hypot(event.xdata - x0, event.ydata - y0)
            self.current_circle.radius = radius
            self.fig.canvas.draw()

    def on_release(self, event):
        if event.button != 1 or not self.dragging:
            return
        
        self.dragging = False
        if self.current_circle.radius > 1:  # Minimum radius check
            self.circles.append(self.current_circle)
        else:
            self.current_circle.remove()
        self.current_circle = None
        self.fig.canvas.draw()

    def on_key_press(self, event):
        if event.key == 'm':
            self.save_mask()
        elif event.key == '1':
            self.mode = "add"
            print("Mask addition mode")
        elif event.key == '2':
            self.mode = "remove"
            print("Mask removal mode")
        elif event.key == 'C':  # Clear all!
            for c in self.circles:
                c.remove()
            self.circles = []
            self.fig.canvas.draw()
            print("All masks removed")
        elif event.key == '=':
            self.vmax = min(self.vmax + 0.2, 100)
            self.update_display()
        elif event.key == '-':
            self.vmax = max(self.vmax - 0.2, 0)
            self.update_display()
        elif event.key == '9':
            self.vmin = max(self.vmin - 0.2, 0)
            self.update_display()
        elif event.key == '0':
            self.vmin = min(self.vmin + 0.2, 100)
            self.update_display()
        elif event.key == ']':
            self.stretch_factor *= 1.5
            self.update_display()
        elif event.key == '[':
            self.stretch_factor /= 1.5
            self.update_display()

    def load_existing_mask(self, mask):
        """Convert binary mask to circle patches"""
        from skimage.measure import find_contours
        
        contours = find_contours(mask, 0.5)
        for contour in contours:
            # Fit circles to mask regions
            y, x = contour[:,0], contour[:,1]
            center_x, center_y = np.mean(x), np.mean(y)
            radius = np.mean(np.sqrt((x-center_x)**2 + (y-center_y)**2))
            
            circle = Circle(
                (center_x, center_y),
                radius=radius,
                edgecolor='orange',
                facecolor='none',
                lw=0.5
            )
            self.ax.add_patch(circle)
            self.circles.append(circle)
        
        self.fig.canvas.draw()
        
    def save_mask(self):
        mask = np.zeros_like(self.original_data, dtype=bool)
        y, x = np.indices(self.original_data.shape)
        
        for c in self.circles:
            cx, cy = c.center
            r = c.radius
            # Simple circle equation - no rotation needed
            mask |= (x - cx)**2 + (y - cy)**2 <= r**2
        
        # Save as compressed numpy format
        np.save(self.output_path, mask)
        print(f"Mask saved to {self.output_path} ({(mask.sum()/mask.size*100):.2f}% masked)")

    def __del__(self):
        """Clean up connections when done."""
        self.canvas.mpl_disconnect(self.cid_press)
        self.canvas.mpl_disconnect(self.cid_motion)
        self.canvas.mpl_disconnect(self.cid_release)
        self.canvas.mpl_disconnect(self.cid_key)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python script.py <fits_file>")
        sys.exit(1)

    gname = sys.argv[1].split("_")[0]
    fits_name = f"/home/pku/Astro/FEASTS_SED/data/{gname}/masked/skybgsub_{sys.argv[1]}.fits"
    save_path = f"/home/pku/Astro/FEASTS_SED/data/{gname}/masked/circ_mask_{sys.argv[1]}.npy"  # Changed output name
    creator = CircleMaskCreator(fits_name, save_path)  # Using CircleMaskCreator
    plt.show()



