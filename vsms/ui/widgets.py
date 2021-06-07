import ipyvue as vue
from traitlets import (Dict, Unicode, List, Integer, Bool)

## TODO: these could be auto generated from the vue file.
class MImageGallery(vue.VueTemplate):
    image_urls = List().tag(sync=True)
    ldata = List().tag(sync=True)
    template = Unicode("""
        <m-image-gallery :image_urls='image_urls' :ldata='ldata'  />
    """).tag(sync=True)