import sys
from pathlib import Path
import time
import numpy as np

main_dir = Path(__file__).parent.parent
sys.path.append(str(main_dir))

from src.ffe_parse import FFEParser


if __name__ == '__main__':
    """使用示例"""
    ffe_path = main_dir / ".data/test.ffe"
    # 初始化解析器
    time_start = time.time()
    ffd = FFEParser.parse(ffe_path)
    time_end = time.time()
    print(f"第一次解析耗时：{time_end - time_start:.2f}s")
    # print(FFEParser.parse.cache_info())

    time_start = time.time()
    ffd = FFEParser.parse(ffe_path)
    time_end = time.time()
    print(f"第二次解析耗时：{time_end - time_start:.2f}s")
    print(FFEParser.parse.cache_info())

    # 定义特定的空域范围
    thetas = np.linspace(0, 90, 10)
    phis = np.linspace(0, 360, 10)
    
    # 取出特定频率、特定空域、特定分量的电场的虚部
    ff = ffd.electric_field.sel(Frequency=2e5, Theta=thetas, Phi=phis)['Etheta'].values.imag
    print(ff)
