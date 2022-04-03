# ydata-viz-ai
Student project for YDATA for Viz.ai

\# may need to install gdcm
conda install -c conda-forge gdcm

# run bokeh
# open terminal in your git directory
# run like this:
bokeh serve dicom_interactive_viewer.py --args <dir with CT images>
# all CT images should be already converted to .npz (numpy array)
bokeh can be installed with conda if missing
