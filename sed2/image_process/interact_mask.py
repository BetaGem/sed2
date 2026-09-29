'''
Interactive circular masking.

Draw circular masks with the mouse on each band and save them to
``<workdir>/masked/circ_mask_<galaxy>_<band>.npy``.  The masks are picked up
automatically by :func:`sed2.image_process.do_mask` through
``load_circ_mask``.

The tool is meant to run **after** the automatic ``mask`` step: it shows the
auto-masked image (``<workdir>/masked/<galaxy>_<band>.fits``) so that the user
can circle sources that survived the automatic masking.  When a mask is saved,
the automatic masking/interpolation is re-run for that band (``apply=True``)
so the manual circles are folded into the cleaned image immediately.

Run it as a pipeline step (a display is required)::

    python demo.py demo.par          # with ``steps ... mask imask ...``

or standalone for a subset of bands::

    python -m sed2.image_process.interact_mask demo.par sdss_g sdss_r

Keys: 1 = add, 2 = remove, m = save, Q = quit, C = clear all,
=/- = vmax, 9/0 = vmin, [ / ] = stretch.
'''  # noqa: EXE002

from contextlib import suppress
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.visualization import AsinhStretch, ImageNormalize
from matplotlib.patches import Circle

__all__ = ["CircleMaskCreator", "interactive_mask"]


class CircleMaskCreator:
    def __init__(self, fits_file, output_path, stretch_factor=0.1):
        import matplotlib.pyplot as plt

        self.fits_file = fits_file
        self.output_path = output_path
        self.original_data = fits.getdata(fits_file)
        self.mode = "add"
        self.vmin = 5.
        self.vmax = 99.8
        self.stretch_factor = stretch_factor

        # Create figure with adaptive size
        aspect_ratio = self.original_data.shape[1] / self.original_data.shape[0]
        figsize = (10 * aspect_ratio, 10)

        self.fig = plt.figure(figsize=figsize)
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
        self.saved = False
        self.quit = False

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
        self.ax.clear()

        # Redraw image
        self.ax.imshow(self.original_data, origin='lower',
                       cmap='gray', norm=self.norm)
        self.ax.text(0.01, 0.01, str(self.fits_file),
                     transform=self.ax.transAxes, fontsize=8,
                     bbox={'fc': 'w', 'alpha': .7})
        self.ax.text(0.99, 0.01, "1:Add | 2:Remove | m:Save | C:ClearMask | Q:QuitAll",
                     transform=self.ax.transAxes, ha='right', va='bottom',
                     fontsize=8, bbox={'fc': 'w', 'alpha': .7})
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
                dist = np.hypot(dx, dy) / c.radius  # Simplified distance check
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
        if event.key == 'Q':
            self.quit = True
            print("Quitting interactive masking ...")
            self.close()
        elif event.key == 'm':
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
            y, x = contour[:, 0], contour[:, 1]
            center_x, center_y = np.mean(x), np.mean(y)
            radius = np.mean(np.sqrt((x - center_x)**2 + (y - center_y)**2))

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
        self.saved = True
        print(f"Mask saved to {self.output_path} ({(mask.sum()/mask.size*100):.2f}% masked)")

    def close(self):
        """Disconnect event handlers and close the figure (idempotent)."""
        import matplotlib.pyplot as plt

        for name in ("cid_press", "cid_motion", "cid_release", "cid_key"):
            cid = getattr(self, name, None)
            if cid is None:
                continue
            self.canvas.mpl_disconnect(cid)
            setattr(self, name, None)
        plt.close(self.fig)

    def __del__(self):
        """Clean up connections when done."""
        with suppress(Exception):
            self.close()


def _input_path(path, galaxy, band, image_type):
    if image_type == "cropped":
        return path / "cropped" / f"crop_{galaxy}_{band}.fits"
    if image_type == "masked":
        return path / "masked" / f"{galaxy}_{band}.fits"
    if image_type == "image":
        return path / "image" / f"skybgsub_{galaxy}_{band}.fits"
    raise ValueError("imask_image should be 'cropped', 'masked' or 'image', "
                     f"got '{image_type}'.")


def interactive_mask(workdir, galaxy, filters, image_type="masked",
                     ref_band=None, overwrite=False, stretch_factor=0.1,
                     apply=True, mask_kwargs=None):
    """
    Interactively draw circular masks on the auto-masked images.

    The masks are written to ``<workdir>/masked/circ_mask_<galaxy>_<band>.npy``
    and consumed by :func:`sed2.image_process.do_mask`.

    Parameters
    ----------
    workdir : str or Path
        The working directory.
    galaxy : str
        Galaxy name.
    filters : list of str
        Bands to mask.
    image_type : str, optional
        Which image to display: ``masked`` (default, the auto-masked image),
        ``cropped`` or ``image``.
    ref_band : str, optional
        If given, only this band is masked (e.g. draw on one reference band).
    overwrite : bool, optional
        Redraw bands that already have a circular mask.
    stretch_factor : float, optional
        Asinh stretch factor for the display.
    apply : bool, optional
        If True, re-run the automatic masking/interpolation for a band right
        after its circular mask is saved, so the cleaned image is updated.
    mask_kwargs : dict, optional
        Keyword arguments forwarded to :func:`do_mask` when ``apply`` is True.
    """
    import matplotlib.pyplot as plt

    # Prefer an interactive backend; fall back to the default if unavailable.
    with suppress(ImportError):
        plt.switch_backend("Qt5Agg")

    path = Path(workdir)
    masked_dir = path / "masked"
    masked_dir.mkdir(parents=True, exist_ok=True)

    if ref_band is not None:
        filters = [ref_band]

    for band in filters:
        in_file = _input_path(path, galaxy, band, image_type)
        if not in_file.is_file():
            print(f"Input image not found: {in_file}, skipping {band}.")
            continue

        out_file = masked_dir / f"circ_mask_{galaxy}_{band}.npy"
        if out_file.is_file() and not overwrite:
            print(f"Circular mask already exists for {band}: {out_file}")
            print("  (set imask_overwrite True to redraw it)")
            continue

        print(f"\n=== Interactive masking: {band} ===")
        print("  1:Add  2:Remove  m:Save  C:ClearMask  Q:QuitAll")
        print("  (close the window for the next band)")
        creator = CircleMaskCreator(in_file, out_file, stretch_factor=stretch_factor)
        plt.show()
        creator.close()

        # fold the manual circles into the cleaned image
        if apply and creator.saved:
            from .image_mask import do_mask
            print(f"Applying manual mask for {band} ...")
            do_mask(path, galaxy, band, **(mask_kwargs or {}))

        # 'Q' in the window ends the loop early
        if creator.quit:
            print("Interactive masking ended by user.")
            break


if __name__ == "__main__":
    import sys

    from ..utils import load_filters, read_par

    if len(sys.argv) < 2:
        print("Usage: python -m sed2.image_process.interact_mask "
              "<par_file> [band ...]")
        sys.exit(1)

    params = read_par(Path(sys.argv[1]))
    workdir = Path(params["workdir"])
    galaxy = params["galaxy"]
    filters = sys.argv[2:] if len(sys.argv) > 2 else \
        load_filters(workdir, gal_name=galaxy)

    mask_kwargs = {
        "bkg_ref_band": params["bkg_ref_band"],
        "crop_size": params.get("mask_box", 3),
        "crop_size_large": params.get("mask_box_large", 6),
        "thresh": params.get("mask_in_thresh", 1.5),
        "dilate": params.get("mask_in_dilate", 1.5),
        "thresh_out": params.get("mask_out_thresh", 3.0),
        "clean": params.get("mask_clean", True),
    }

    interactive_mask(workdir, galaxy, filters,
                     image_type=params.get("imask_image", "masked"),
                     ref_band=params.get("imask_band", None),
                     overwrite=params.get("imask_overwrite", False),
                     apply=params.get("imask_apply", True),
                     mask_kwargs=mask_kwargs)
