import ipyvue as vue
from traitlets import (Dict, Unicode, List, Integer, Bool)

## TODO: these could be auto generated from the vue file.
class MImageGallery(vue.VueTemplate):
    image_urls = List().tag(sync=True)
    ldata = List().tag(sync=True)
    # modal doesn't work well within notebook, hard to close, so turn off in widget
    template = Unicode("""
        <m-image-gallery :image_urls='image_urls' :ldata='ldata' :with_modal='false' />
    """).tag(sync=True)