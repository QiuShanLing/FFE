import sys
import os
import time
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from FFE import FFEParser


if __name__ == '__main__':
    """使用示例"""
    ffe_path = os.path.join(os.path.dirname(__file__), r'../.data/ffdata.ffe')
    # 初始化解析器
    time_start = time.time()
    ffd = FFEParser.parse(ffe_path)
    time_end = time.time()
    print(f"第一次解析耗时：{time_end - time_start:.2f}s")
    print(FFEParser.parse.cache_info())

    time_start = time.time()
    ffd = FFEParser.parse(ffe_path)
    time_end = time.time()
    print(f"第二次解析耗时：{time_end - time_start:.2f}s")
    print(FFEParser.parse.cache_info())

    # 定义特定的空域范围
    thetas = np.linspace(0, 180, 19)
    phis = np.linspace(0, 360, 1)
    
    # 取出特定频率、特定空域、特定分量的电场的虚部
    ff = ffd.electric_field.sel(Frequency=2e5, Theta=thetas, Phi=phis)['Etheta'].values.imag
    print(ff)
