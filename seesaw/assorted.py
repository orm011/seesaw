import PIL.Image
from IPython.display import HTML

def show_image_box(url, box_rel):
    (left, top, right, bottom) = [round(b*100) for b in box_rel]
    return HTML(
    f'''
    <div
        style='position: relative;'
    >
        <img 
            src='{url}'
            style='object-fit: contain; 
                    position: relative; 
                    opacity: .3;'
        >
        <img 
            src='{url}'
            style='object-fit: contain; 
                    position: absolute; 
                    left: 0; 
                    top: 0; 
                    margin: 0;
                    clip-path: polygon( 
                                    {left}% {top}%, 
                                    {right}% {top}%, 
                                    {right}% {bottom}%, 
                                    {left}% {bottom}%
                                );'
        >
    </div>
    '''
    )

def display_row(ds, dfrow, host='localhost.localdomain', port=10000):
    dbidx = int(dfrow['dbidx'])
    x1=dfrow['x1']; y1=dfrow['y1']; x2=dfrow['x2']; y2=dfrow['y2']
    path = f'{ds.dataset_root}/images/{ds.paths[dbidx]}'
    url = f'http://{host}:{port}/{path}'
    im = PIL.Image.open(path)
    w,h = im.size
    return show_image_box(url, [x1/w, y1/h, x2/w, y2/h])