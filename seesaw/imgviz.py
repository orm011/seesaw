from plotnine import (
    ggplot,
    geom_polygon,
    geom_map,
    facet_wrap,
    aes,
    geom_text,
    annotate,
    scale_y_reverse,
    xlim,
    ylim,
    theme,
    element_blank,
    geom_vline,
    geom_hline,
)
from plotnine import scale_color_discrete
from plotnine import geom_rect, geom_polygon
import matplotlib.pyplot as plt
import io
import PIL
import torchvision.transforms as T  # for rescaling


def theme_image(w, h, dpi=80):
    """defaults for showing images without coordinates"""
    return theme(
        axis_line=element_blank(),
        axis_text_x=element_blank(),
        axis_text_y=element_blank(),
        axis_ticks=element_blank(),
        axis_ticks_length=0,
        axis_ticks_pad=0,
        axis_title_x=element_blank(),
        axis_title_y=element_blank(),
        legend_box=element_blank(),
        legend_position=None,
        plot_margin=0,
        panel_grid=element_blank(),
        plot_background=element_blank(),
        panel_background=element_blank(),
        panel_border=element_blank(),
        panel_grid_major=element_blank(),
        panel_grid_minor=element_blank(),
        figure_size=(w / dpi, h / dpi),
        dpi=dpi,
    )


def ggimg(image, mapping=None, data=None, dpi=80):
    w, h = image.size
    return (
        ggplot(mapping=mapping, data=data)
        + scale_y_reverse(limits=(0, h))
        + xlim(0, w)
        + scale_color_discrete(guide=False)  # removes legend for line color
        + theme_image(w, h, dpi=dpi)
        + annotate(
            "rect", xmin=0, xmax=w, ymin=0, ymax=h, color="black", fill=None
        )  # box around image
    )


def rescale_data(im, bx, box_cols=["xmin", "xmax", "ymin", "ymax"], max_size=200):
    tx = T.Resize(max_size)
    w, h = im.size
    im2 = tx(im)
    w2, h2 = im2.size
    sf = w2 / w
    for c in box_cols:
        bx2 = bx.assign(**{c: bx[c] * sf})
    return im2, bx2


def add_image(f, im):
    w, h = im.size
    for ax in f.get_axes():
        ax.imshow(im, origin="lower", extent=(0, w, 0, -h))
        ax.set_xlim(0, w)
        ax.set_ylim(-h, 0)
    return f


def draw_ggimg(ggim):
    f = ggim.draw()
    f = add_image(f, ggim.environment.eval("image"))
    return f


def ggimg_draw(ggim):
    return draw_ggimg(ggim)


def add_mask(f, mask, d=96):
    h, w = mask.shape([0, 1])
    for ax in f.get_axes():
        ax.matshow(mask, origin="lower", extent=(0, d * w, 0, -d * h), alpha=0.2)
        ax.set_xlim(0, d * w)
        ax.set_ylim(-d * h, 0)
    return f


def toPIL(gimg):
    plt.ioff()
    try:
        buf = io.BytesIO()
        f = draw_ggimg(gimg)
        f.savefig(buf, format="png")
        plt.close(f)
        buf.seek(0)
        im = PIL.Image.open(buf)
    finally:
        plt.ion()

    return im
