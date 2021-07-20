import csv

'''预先加载libsixel。'''
from ctypes import cdll
cdll.LoadLibrary('/usr/local/lib/libsixel.so')

def read_text_file(filename):
    '''读取文本文件的完整内容。

    Args:
        filename: 文件路径，推荐使用完整，因为不同的系统上当前目录不同。

    Returns:
        字符串形式的文件完整内容。
    '''
    with open(filename) as f:
        return f.read()

def read_sixel_image(filename):
    '''读取Sixel格式图片的完整内容。

    Args:
        filename: 图片文件路径。

    Returns:
        图片文件内容以及一个换行符前缀。
    '''
    return '\n' + read_text_file(filename)

def read_csv_records(filename):
    '''读取CSV文件的数据记录（忽略标题）

    Args:
        filename: CSV文件路径。

    Returns:
        二维表形式的数据集，所有数据是字符串形式
    '''
    with open(filename) as f:
        reader = csv.reader(f)
        next(reader)
        return [row for row in reader]
