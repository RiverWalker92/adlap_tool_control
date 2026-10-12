"""Write complete artifact bytes to a temporary file, validate, then publish."""
import os,tempfile
from pathlib import Path


def atomic_write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix=path.name+'.',suffix='.part',dir=path.parent)
    temporary=Path(name)
    try:
        with os.fdopen(fd,'wb') as stream:
            written=stream.write(data)
            if written!=len(data):raise IOError('Incomplete artifact write: '+str(path))
            stream.flush();os.fsync(stream.fileno())
        if temporary.read_bytes()!=data:raise IOError('Artifact byte verification failed: '+str(path))
        temporary.replace(path)
    finally:
        if temporary.exists():temporary.unlink()


def save_figure(fig,path):
    import io
    import matplotlib.pyplot as plt
    from PIL import Image
    path=Path(path)
    png=io.BytesIO();fig.savefig(png,format='png',dpi=220,bbox_inches='tight');png_data=png.getvalue()
    with Image.open(io.BytesIO(png_data)) as image:image.load()
    atomic_write(path.with_suffix('.png'),png_data)
    plt.close(fig)
