# Tacta Counter - AI Coding Agent Instructions

## Project Overview
Computer vision system for automatically counting game points in Tacta by detecting and classifying colored circles using OpenCV, k-means clustering, and HSV color space analysis.

**Game Context**: In Tacta, white dots (visible game pieces) on colored cards represent points for the player who owns that color. This tool counts the final point distribution after a game by detecting white circles and identifying the card colors beneath them.

## Architecture & Workflow

### Code Organization
- **`tacta.py`**: Core image processing and color analysis functions (production code)
- **`tacta_counter.ipynb`**: Interactive exploration and visualization notebook

### Main Processing Pipeline
1. **Image loading** → `tacta.load_image()` returns RGB, grayscale, and HSV versions
2. **Adaptive thresholding** → `tacta.apply_adaptive_threshold()` with Gaussian blur to detect edges
3. **Contour detection** → `tacta.find_contours()` via `cv2.findContours`
4. **Circle filtering** → `tacta.filter_circular_contours()` selects circles by diameter (8-18px), intensity, std deviation, and saturation. Use `return_params=True` to get filtering metrics for analysis.
5. **Donut region extraction** → `tacta.extract_donut_colors()` samples colors from ring around each circle (radius+2 to radius+6 pixels)
6. **HSV to 3D Cartesian** → `tacta.convert_hsv_to_cartesian_array()` for visualization
7. **K-means clustering** → Group detected colors into clusters (typically 6-7: one per game color + background)
8. **Cluster-to-color mapping** → `tacta.assign_clusters_to_colors()` greedy assignment by Euclidean distance in HSV space

### Critical Color Space Transformations
- **HSV ranges**: Hue 0-179, Saturation 0-255, Value 0-255 (OpenCV convention)
- **Cartesian mapping**: `hsv_to_cartesian()` in `tacta.py` converts to (x, y, z) where:
  - x = (h/180) * 2π radians
  - y = s/255 (normalized saturation)
  - z = v/255 (normalized value)
- Use cylindrical coordinates for 3D plotting: x = radius*cos(angle), y = radius*sin(angle), z = height

## Development Environment

### Setup
```bash
conda env create -f env.yml  # Creates 'tacta' environment with Python 3.13
conda activate tacta
```

### Key Dependencies
- **opencv** (cv2) - Image processing and contour detection
- **matplotlib** + **ipympl** - Use `%matplotlib widget` for interactive plots in notebooks
- **plotly** + **anywidget** - Interactive 3D scatter plots with click handlers
- **scikit-learn** - K-means clustering
- **numpy** - Array operations

### Running the Notebook
Execute cells sequentially from top to bottom. Critical interactive features:
- Click points in 3D Plotly scatter plots to view cropped circle regions
- Adjust filtering thresholds: `min_diameter`, `max_diameter`, `max_std_dev_threshold`, `max_saturation`

## Project-Specific Conventions

### Code Structure
- **Keep notebook lean**: Use `tacta.py` functions for processing logic, notebooks for exploration and visualization
- **Type hints**: Use `NDArray` from `numpy.typing` for array parameters (without strict dtype specifications to avoid OpenCV type conflicts)
- **Type ignore comments**: Acceptable for OpenCV conversions that have complex type signatures

### Image Data
- Input images in `data/` directory (current: `cropped_black.jpg`)
- All images converted from BGR (OpenCV default) to RGB for matplotlib display
- Darkened visualization: `tacta.darken_image(image_rgb, factor=0.6)` for better contour visibility

### Filtering Logic
Circle detection requires ALL conditions met (parameters for `tacta.filter_circular_contours()`):
- `circularity > 0.4` (perimeter²/area threshold)
- `min_radius <= radius <= max_radius` (default: 4-9px for diameter 8-18)
- `mean_intensity > 100` (white/gray fill detection)
- `std_intensity < max_std_dev_threshold` (default: 20, uniform fill)
- `mean_saturation < max_saturation` (default: 80, low color in center)

### Reference Colors (HSV values)
Hard-coded in notebook at line ~400, measured from example game images:
```python
reference_colors = {
    "Red": (0, 200, 200),
    "Yellow": (30, 200, 200),
    "Green": (60, 200, 200),
    "Blue": (120, 200, 200),
    "Pink": (170, 200, 200),
    "White": (0, 0, 200)
}
```
**Calibration**: These HSV values were sampled from actual game cards. If working with new images under different lighting:
1. Run notebook up to the 3D scatter plot (cell ~365-519)
2. Click on correctly-detected circles in the Plotly visualization
3. Note the HSV values from the displayed regions
4. Update `reference_colors` dictionary with observed cluster centers

## Common Operations

### Adding New Color References
Modify `reference_colors` dictionary with HSV tuples. Cluster assignment uses greedy nearest-neighbor matching, so order matters when clusters are equidistant.

### Adjusting Detection Sensitivity
- **More circles detected**: Increase `max_diameter`, decrease `min_diameter`, increase `max_saturation`, increase `max_std_dev_threshold`
- **Fewer false positives**: Decrease `max_diameter`, increase `min_intensity` threshold, decrease `max_saturation`

**Adapting to different lighting/camera conditions**:
- **Darker images**: Decrease `min_intensity` threshold (currently 100) to detect dimmer white circles
- **High contrast/shadows**: Increase `max_std_dev` to allow more intensity variation within circles
- **Colored lighting**: Adjust `max_saturation` threshold if white circles appear tinted
- **Different resolutions**: Scale `min_diameter`/`max_diameter` proportionally (currently 8-18px for the example image)
- **Verify changes**: Re-run filtering with new parameters and check circle count

### Debugging Circle Detection
The notebook visualization cells show filtered contours overlaid on the darkened image. Check the count of detected circles and visually inspect if the green contours match the expected white dots.

**Parameter analysis**: Use `return_params=True` in `tacta.filter_circular_contours()` to get a dictionary with 'diameter', 'circularity', 'mean_intensity', 'std_intensity', and 'mean_saturation' for all filtered circles. This helps visualize the actual distribution of detected circles and tune thresholds effectively.

## Git Workflow

### Notebook History Cleanup
`clean_history.sh` uses `git filter-repo` + `nbstripout` to remove notebook outputs from git history:
```bash
./clean_history.sh  # Interactive prompt before force push
```
**Warning**: Rewrites history and force-pushes. Use only when outputs need removal.
