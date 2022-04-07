# ydata-viz-ai
Student project for YDATA for Viz.ai

\# may need to install gdcm
conda install -c conda-forge gdcm

# run bokeh
1. open terminal in project directory
2. prepare directory with CT images converted to .npz
3. run like this:
bokeh serve dicom_interactive_viewer.py --args <dir path with CT images>
4. there will be a link in cmd output. open this link in browser

bokeh can be installed with conda if missing
