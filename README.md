# 3D Color Package: End-to-End Workflow with InterDeCA

This tutorial guides you through the complete workflow of the **3D Color Package** (InterDeCA), from installation to analyzing color variation and performing mesh segmentation. We will use a **10-mussel dataset** as our example.

## Prerequisites

- **3D Slicer**: Download and install from [download.slicer.org](https://download.slicer.org/).
- **Dataset**: Download the 10-mussel dataset and extract it to a local directory (e.g., `Tutorials/data`).

---

## Part 1: Installation

This installation covers how to download and install the **InterDeCA** tool. We assume you already have the 3D Slicer software downloaded and installed on your system (if you don't, find install options [here](https://download.slicer.org/)). 

### Downloading

#### InterDeCA Module and ATLAS Dependency

Open your system's command-line terminal, and run the following command:
```bash
git clone --branch demo --recurse-submodules https://github.com/bshi42/3d_color_package.git
```
The GIF below depicts what this process should look like. Please note, you likely will not need to sign in to clone the repo after the software is published.

![A GIF showing the InterDeCA + ATLAS download process.](./Tutorials/images/download_interdeca_atlas.gif)

### Installing in 3D Slicer

#### ATLAS 

You've already downloaded ATLAS as a submodule of InterDeCA. However, you will still need to install it in 3D Slicer separately. Following the instructions provided in the [ATLAS github repository](https://github.com/agporto/ATLAS), perform the following steps:
1. Add the cloned top-level `ATLAS` folder to Slicer using the dropdown menu: `Developer Tools -> Extension Wizard -> Select Extension -> Select ATLAS`
2. Open one of the ATLAS modules (BUILDER, DATABASE, PREDICT, or SEGMENTATION) to confirm it loaded correctly.

The GIF below demonstrates what this process should look like.

![A GIF showing the ATLAS installation process.](./Tutorials/images/install_atlas.gif)

#### InterDeCA

First, open the Python console in 3D Slicer and perform the following commands:

```python
>>> import slicer
>>> slicer.util.pip_install('imageio')
>>> slicer.util.pip_install('scikit-learn')
>>> slicer.util.pip_install('umap-learn')
>>> slicer.util.pip_install('scikit-image')
```

![The 3D Slicer top menu, with the Python console button outlined in red.](./Tutorials/images/python_console_3dslicer.png)

The image above shows where the Python console button is in the 3D Slicer UI (the console will open on the bottom right of the UI). Once this is done, restart the 3D Slicer app.

Next, perform the following steps in order:
1. Go to `Edit -> Application Settings -> Modules` in 3D Slicer.
2. Add the path to `color_deca/deca3` directory within the downloaded `3d_color_package` repository by dragging and dropping the `deca3` folder from your file system.
3. Restart 3D Slicer.

The GIF below shows how to perform steps 1 and 2.

![A GIF showing steps 1 and 2 of the above procedure for installing InterDeCA](./Tutorials/images/add_interdeca.gif)

#### Blender (Optional)

InterDeCA also includes auto-detection and auto-install functionality for Blender, which we use for texture baking functionality. If it doesn't work on your system, Blender can be downloaded and installed manually following the resources [here](https://www.blender.org/download/).


---

## Part 2: Generating an Atlas (ATLAS Tab)

### Overview

There are 3 main steps to run ATLAS and texture transfer:
1. **Model Import:** Select any of the 3D models to be used in generating the atlas and import it into Slicer.
2. **Hyperparameter Tuning:** Tune the UV mapping and texture baking parameters (if necessary).
3. **Run:** Click the `Run ATLAS and Texture Transfer` button.

### Model Import

This can be done in the `Welcome to Slicer` menu by clicking the `Add Data` button. After clicking, there will be a prompt to find and select data files to import from your file system. For running ATLAS and texture transfer, selecting any of the 3D model (.obj) files in your desired dataset will suffice.

### Hyperparameter Tuning

After importing one of your 3D models, open InterDeCA from Slicer's module selector. The module selector is the drop-down near the top of the Slicer window that may currently say `Welcome to Slicer`; type `InterDeCA` in that selector's search box, or browse to the category `SlicerMorph -> DeCA Toolbox -> InterDeCA`.

This is not a top-level application menu. If the module selector shows `ATLAS` but does not show `SlicerMorph` or `InterDeCA`, ATLAS is loaded but InterDeCA is not. Confirm that the `color_deca/deca3` folder was added in `Edit -> Application Settings -> Modules`, restart Slicer, and check the Python console or application log for module-loading errors.

![The InterDeCA module's ATLAS tab menu.](./Tutorials/images/atlas_module_menu.png)

First, you'll need to enter the paths to the directories in your file system containing the desired models, landmarks, and textures. You'll also need to provide an output directory where all the produced files will be stored; we recommend using an empty directory so that the results from different runs don't get mixed up.

If you've already run ATLAS, you can enter in the paths to the generated atlas model and atlas landmarks. This is optional.

Under `Blender (cleanup, UV, bake)`, you'll see different hyperparameters. The first is the filepath to the blender executable on your system. If running on Linux, for guaranteed stability, we recommend pre-installing Blender using the command
```bash
sudo snap install blender --classic
```
and entering in the resultant filepath. For Windows and macOS users, InterDeCA will attempt to auto-install Blender if it isn't detected in your file system. 

The `Merge by distance` hyperparameter specifies the distance below which vertices are automatically merged; this cleans up duplicate/overlapping vertices before UV unwrapping.

The `Smart UV angle (deg)` hyperparameter specifies the maximum angle between faces that can be included in the same UV island; this controls how the 3D surface is "cut" and flattened into 2D UV space. 

The `Island margin (UV)` hyperparameter specifies the spacing between UV islands in the 0-1 UV space to prevent texture bleeding between different parts of the model. 

The `Bake size (px)` hyperparameter specifies the resolution of the output texture map (px by px). The higher this value, the more detailed the texture map, but the larger the file size.

The `Bake extrusion` hyperparameter specifies how far to "push out" the baked data from the surface in order to help capture details while preventing gaps in the texture.

The `Bake margin (px)` hyperparameter specifies the pixel padding around UV islands in the baked texture. This prevents edge artifacts and seams.

In some cases, resultant textures will have black speckles; we've found decreasing the `Merge by distance` value by a few orders of magnitude significantly helps. Otherwise, the default hyperparameter values (usually) work well.

### Run

Once all the necessary fields are filled-out and the hyperparameters are tuned, click the green `Run ATLAS and Texture Transfer` button at the bottom of the ATLAS tab menu. This will run atlas generation and texture transfer using the input sets of models, landmarks, and textures.

---

## Part 3: Analyzing Color (MultiRecolor Tab)

The **MultiRecolor** tab allows you to analyze and visualize the color variation across your population using the aligned textures generated in Part 2. The workflow consists of 4 steps:

### Step 1: Multi-texture Clustering
This step processes all texture images to create a consistent, shared color palette for comparison.
1. **Model**: Select the generated Atlas model (from your output folder).
2. **Texture Directory**: Select the directory containing the *transferred* textures (usually in your output folder under `atlas_textures`).
3. **Mode**: Choose **Clustering** to create a shared palette.
4. **Parameters**:
   - `Initial Clusters`: 24 (higher for more detail).
   - `Consolidated Clusters`: 8 (number of final colors).
5. Click **Cluster**.

### Step 2: Individual Visualization
After clustering, it is crucial to visualize how the shared palette represents each individual specimen. This confirms that the color simplification (quantization) accurately captures the main patterns of the original specimen.

1. **Select Texture**: Choose a specific texture file from the dropdown list.
2. **Apply Texture**: Click to apply the clustered texture to the 3D model.
3. **Toggle Raw/Clustered**: You can compare the original (raw) texture vs. the clustered result to ensure the color details are preserved.
   - If the result looks "patchy" or misses key details, consider increasing the number of `Consolidated Clusters` in Step 1.

### Step 3: Population Analysis
This step performs dimensionality reduction to compare color patterns across all textures in a scatter plot.
1. **Method**: Select **PCA** (Principal Component Analysis).
2. **Number of PCs**: 2.
3. Click **Compare Textures**.
4. A scatter plot will appear. Each point represents one mussel specimen.
   - Points closer together have more similar color patterns.
   - Points farther apart are more distinct.

<img src="./Tutorials/images/PCA.png" alt="PCA Analysis of the 10-mussel dataset" width="800"/>

### Step 4: Morphospace
This step allows you to interactively explore the color variation by navigating through the simplified color space (PCA space).
1. Expand the **Morphospace** section.
2. Click **Visualize Morphospace**.
3. Use the **X-axis** and **Y-axis** sliders to explore color variations.
   - **Interactive Exploration**: As you move the sliders, the 3D model updates in real-time to show the *predicted* color pattern for that specific position in the PCA space.
   - You can visualize hypothetical transitions between different color morphs (e.g., seeing how a striped pattern might fade into a solid color).

<img src="./Tutorials/images/morphospace.gif" alt="Morphospace Dragging X-Axis" width="100%"/>




---

## Part 4: Segmentation (Mesh Region Selection)

This tutorial covers the **Mesh Region Selection** tab in InterDeCA, which allows you to interactively segment a Region of Interest (ROI) on an atlas model using landmark points or closed curves. This is useful for isolating specific biological structures or regions for analysis.

### Creating a Segmentation

1.  **Navigate to the Mesh Region Selection Tab**
    *   Click on the **Mesh Region Selection** tab in the InterDeCA module interface.

    <img src="./Tutorials/images/DeCA_Segmentation_1.png" width="500">

2.  **Select Target Mesh**
    *   In the **Target Mesh** dropdown, select the model you wish to segment.

3.  **Choose Selection Method**
    *   In the **Selection Markup** dropdown, select **Create new Closed Curve** (recommended) or **Create new Point List**.
        *   **Closed Curve**: Automatically connects the points with a line, making it easier to visualize the boundary of your region.
        *   **Point List**: Places individual points (fiducials). The region will be defined by the area enclosed by these points, but the boundary line won't be explicitly drawn.

4.  **Define the Region Boundary**
    *   Click the **Draw Landmarks** button.
    *   The cursor will change to a crosshair. Click on the surface of your atlas model to place points that define the boundary of your region.
        *   If no markup node is selected, InterDeCA will automatically create a new "SelectionCurve" (Closed Curve) for you.
        *   Continue placing points to trace the outline of the region you want to select.
    *   **Press ESC** on your keyboard when you are finished placing points to exit the drawing mode. The curve will automatically close.

    <img src="./Tutorials/images/DeCA_Segmentation_Draw.png" width="500">

    > **Note:** You can adjust the position of any point after drawing by clicking and dragging it on the 3D view.

5.  **Configure Selection Options**
    *   **Select only one side**:
        *   **Checked (Default)**: Restricts the selection to vertices that are on the "same side" as your landmarks. This is useful for selecting a patch on a closed surface without selecting vertices on the opposite side of the mesh (e.g., selecting the face of a skull without selecting the back of the head).
        *   **Unchecked**: Selects all vertices inside the boundary curve, potentially projecting through the mesh if the geometry is complex.

6.  **Apply Selection**
    *   Click the **Apply Landmark Selection** button.
    *   The selected region will be highlighted (typically in red) on the atlas model.
    *   The info box will update to show how many vertices were selected and the percentage of the total mesh.

    <img src="./Tutorials/images/DeCA_Segmentation_Apply.png" width="500">

7.  **Refine (Optional)**
    *   If the selection isn't quite right, you can move the landmark points and click **Apply Landmark Selection** again to update the result.
    *   To start over, click the **Clear Selection** button.

### Exporting the Segmented Region

Once you are satisfied with the selection, you can export it as a new, separate model.

8.  **Name the Region**
    *   Enter a name for the new model in the **Export Name** field (default is "SelectedRegion").

9.  **Export**
    *   Click **Export Selected Region as Model**.
    *   A new model node containing only the selected vertices and faces will be added to the scene. You can now save this model or use it for further analysis.



Run slicer headless

/Applications/Slicer.app/Contents/MacOS/Slicer  --no-splash --no-main-window --python-script "/Users/eric/code/3d_color_package/tests/slicer/main.py


https://www.slicer.org/wiki/Documentation/Nightly/Developers/Python_scripting#How_to_run_Python_script_using_a_non-Slicer_Python_environment