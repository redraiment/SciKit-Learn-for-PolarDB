from libsixel.encoder import Encoder, SIXEL_OPTFLAG_INPUT, SIXEL_OPTFLAG_OUTPUT
from pgsklearn.io import read_sixel_image
import matplotlib.pyplot as plt
import seaborn as sns

PNG_FILE_PATH = '/home/postgres/images/plot.png'
SIXEL_FILE_PATH = '/home/postgres/images/plot.sixel'

sns.set_theme()

def as_sixel(canvas):
    canvas.savefig(PNG_FILE_PATH)
    encoder = Encoder()
    encoder.setopt(SIXEL_OPTFLAG_OUTPUT, SIXEL_FILE_PATH)
    encoder.encode(PNG_FILE_PATH)
    plt.close()
    return read_sixel_image(SIXEL_FILE_PATH)

def scatter(data, x, y, hue):
    '''
    生成sns#scatterplot，并返回sixel格式图片。
    '''
    return as_sixel(sns.scatterplot(data=data, x=x, y=y, hue=hue).figure)

def line(data, x, y, hue):
    '''
    生成sns#lineplot，并返回sixel格式图片。
    '''
    return as_sixel(sns.lineplot(data=data, x=x, y=y, hue=hue).figure)

def lm(data, x, y, hue, col, row, height, aspect, scatter, fit_reg):
    '''
    生成sns#lmplot图，并返回sixel格式图片。
    '''
    return as_sixel(sns.lmplot(data=data, x=x, y=y, hue=hue, col=col, row=row, height=height, aspect=aspect, scatter=scatter, fit_reg=fit_reg))
