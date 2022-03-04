import dicom_utils as dcm
import matplotlib.pyplot as plt
import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Slider, RangeSlider, TextInput
from bokeh.plotting import figure, show


def plot_dicom_img_slice(axis, slice_idx, cmap='gray'):
    img3d = ct_img   # debug
    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    if axis == 0:
        plt.imshow(img3d['data'][slice_idx, :, :].T, cmap=cmap)
    elif axis == 1:
        plt.imshow(img3d['data'][:, slice_idx, :], cmap=cmap)
    elif axis == 2:
        plt.imshow(img3d['data'][:, :, slice_idx], cmap=cmap)
    ax.set_aspect(img3d['aspect'][axis])


def plot_dicom_img_slice_bokeh(img3d, axis, slice_idx, bokeh_plot, palette='Greys9'):
    if axis == 0:
        bokeh_plot.image(
            image=[img3d['data'][slice_idx, :, :].T],
            x=0, y=0, dw=10, dh=10,
            palette=palette, level="image"
        )
    elif axis == 1:
        bokeh_plot.image(
            image=[img3d['data'][:, slice_idx, :]],
            x=0, y=0, dw=10, dh=10,
            palette=palette, level="image"
        )
    elif axis == 2:
        bokeh_plot.image(
            image=[img3d['data'][:, :, slice_idx]],
            x=0, y=0, dw=10, dh=10,
            palette=palette, level="image"
        )
    #plot.aspect_ratio = img3d['aspect'][axis]


def update_data(attr, old, new):
    hu_low, hue_high = hu_filter.value
    filtered = dcm.ct_image_filter(ct_img, hu_low, hue_high)
    #plot_dicom_img_slice_bokeh(filtered, 2, slice_idx.value, plot)
    source.data = {'value': [filtered['data'][:, :, slice_idx.value]]}


#if __name__ == "__main__":
path = "./data/CQ500CT1 CQ500CT1/Unknown Study/CT 2.55mm"
ct_img = dcm.read_ct_scan(path)
ct_img = dcm.scale_ct_img(ct_img, (300, 300))

plot = figure()
plot_dicom_img_slice_bokeh(ct_img, 2, 20, plot)

source = ColumnDataSource(data={'value': [ct_img['data'][:, :, 0]]})
plot.image(
    'value', source=source,
    x=0, y=0, dw=10, dh=10,
    palette='Greys9', level="image"
)

slice_idx = Slider(title="slice_idx", value=20, start=0, end=36, step=1, width=500)
slice_idx.on_change('value', update_data)

hu_filter = RangeSlider(title="hue_filter", value=(0, 100), start=-10, end=2000, step=1)
hu_filter.on_change('value', update_data)

inputs = column(slice_idx, hu_filter)
curdoc().add_root(column(inputs, plot, width=800))
curdoc().title = "CT scan"

