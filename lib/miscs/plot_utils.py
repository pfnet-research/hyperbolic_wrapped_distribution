import numpy as np

from PIL import Image

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


def getfig(figsize, xmargin=0.18, ymargin=0.2, xoffset=0, yoffset=0):
    width, height = figsize
    fig = plt.figure(figsize=(
        width / (1 - xmargin * 2), height / (1 - ymargin * 2)
    ))
    fig.subplots_adjust(
        left=(xmargin + xoffset), right=(1 - xmargin + xoffset),
        bottom=(ymargin + yoffset), top=(1 - ymargin + yoffset))
    return fig


def finalize(xlabel=None, ylabel=None, xlabelpad=-12, ylabelpad=-25,
             xrange_=None, yrange_=None):
    if xrange_:
        plt.xticks(xrange_)
        plt.xlim(xrange_)
    else:
        plt.xticks(plt.xlim())
    if yrange_:
        plt.yticks(yrange_)
        plt.ylim(yrange_)
    else:
        plt.yticks(plt.ylim())
    plt.gca().xaxis.set_minor_locator(mpl.ticker.AutoLocator())
    plt.gca().yaxis.set_minor_locator(mpl.ticker.AutoLocator())
    if xlabel:
        plt.xlabel(xlabel, labelpad=xlabelpad)
    if ylabel:
        plt.ylabel(ylabel, labelpad=ylabelpad)


sns.set_style('ticks')
sns.set_context('paper', font_scale=2.0, rc={
    'axes.linewidth': 2.0,
    'lines.linewidth': 2.0,
    'font.family': 'Helvetica'
})


def get_image_array(X, index, image_shape=(1, 28, 28), offset=-0.5):
    if X.dtype == 'uint8':
        ret = X[index]
    else:
        ret = X[index] - offset
        ret = (ret * 255.).clip(0, 255).astype(np.uint8)
    if image_shape[0] == 1:
        ret = ret.reshape(image_shape[1], image_shape[2])
    elif image_shape[0] == 3:
        ret = ret.reshape(image_shape).transpose(1, 2, 0)
    else:
        raise ValueError
    return ret


def get_image_tile(
        data, width, height,
        image_shape=(1, 28, 28), margin=0, offset=-0.5, mode='RGB'):

    shp = image_shape[1:]

    im = Image.new(mode, (shp[0] * width + margin * (width + 1),
                          shp[1] * height + margin * (height + 1)))
    for (x, y), val in np.ndenumerate(np.zeros((width, height))):
        if len(data) <= x + y * width:
            continue
        im.paste(
            Image.fromarray(get_image_array(
                data, x + y * width, image_shape=image_shape, offset=offset)),
            (x * shp[0] + (x + 1) * margin, y * shp[1] + (y + 1) * margin))
    return im
