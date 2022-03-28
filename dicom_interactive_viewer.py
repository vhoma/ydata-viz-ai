import os
import sys
import dicom_utils as dcm
import matplotlib.pyplot as plt
import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Slider, RangeSlider, TextInput, Dropdown
from bokeh.plotting import figure, show


# def plot_dicom_img_slice(axis, slice_idx, cmap='gray'):
#     img3d = ct_img   # debug
#     plt.figure(figsize=(6, 6))
#     ax = plt.gca()
#     if axis == 0:
#         plt.imshow(img3d['data'][slice_idx, :, :].T, cmap=cmap)
#     elif axis == 1:
#         plt.imshow(img3d['data'][:, slice_idx, :], cmap=cmap)
#     elif axis == 2:
#         plt.imshow(img3d['data'][:, :, slice_idx], cmap=cmap)
#     ax.set_aspect(img3d['aspect'][axis])
#
#
# def plot_dicom_img_slice_bokeh(img3d, axis, slice_idx, bokeh_plot, palette='Greys9'):
#     if axis == 0:
#         bokeh_plot.image(
#             image=[img3d[slice_idx, :, :].T],
#             x=0, y=0, dw=10, dh=10,
#             palette=palette, level="image"
#         )
#     elif axis == 1:
#         bokeh_plot.image(
#             image=[img3d[:, slice_idx, :]],
#             x=0, y=0, dw=10, dh=10,
#             palette=palette, level="image"
#         )
#     elif axis == 2:
#         bokeh_plot.image(
#             image=[img3d[:, :, slice_idx]],
#             x=0, y=0, dw=10, dh=10,
#             palette=palette, level="image"
#         )
#     #plot.aspect_ratio = img3d['aspect'][axis]


def show_file(file_path):
    ct_img = np.load(file_path)['I']
    global source
    source.data['value'] = [ct_img[:, :, slice_idx.value]]
    slice_idx.end = ct_img.shape[2]
    return ct_img


def update_file(event):
    global img3d
    img3d = show_file(event.item)
    update_data(None, None, None)
    choose_file.label = event.item


def update_data(attr, old, new):
    # get slice
    global img3d
    ct_slice = img3d[:, :, slice_idx.value]

    # filter
    hu_low, hue_high = hu_filter.value
    filtered = dcm.ct_image_filter(ct_slice, hu_low, hue_high)
    global source
    source.data['value'] = [filtered]


print(sys.argv)
#path = "/Users/boriskefer/Documents/coding/YDATA/2021_2022_main/viz_ai/ydata-viz-ai/data_new/CQ500CT0 CQ500CT0/Unknown Study/CT 4cc sec 150cc D3D on"
path = sys.argv[1]
img_list = [f_name for f_name in os.listdir(path) if f_name.endswith('npz')]

plot = figure()

source = ColumnDataSource(data={'value': []})
plot.image(
    'value', source=source,
    x=0, y=0, dw=10, dh=10,
    palette='Greys9', level="image"
)

slice_idx = Slider(title="slice_idx", value=20, start=0, end=36, step=1, width=500)
slice_idx.on_change('value', update_data)

hu_filter = RangeSlider(title="hue_filter", value=(0, 100), start=-1000, end=1000, step=1)
hu_filter.on_change('value', update_data)

menu = [(name, os.path.join(path, name)) for name in img_list]
choose_file = Dropdown(label=img_list[0], menu=menu)
choose_file.on_click(update_file)

img3d = show_file(os.path.join(path, img_list[0]))

inputs = column(slice_idx, hu_filter, choose_file)
curdoc().add_root(column(inputs, plot, width=800))
curdoc().title = "CT scan"

