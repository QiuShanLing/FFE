from turtle import clear
import numpy as np
import pandas as pd
from typing import List, Tuple
import xarray as xr
from functools import lru_cache
import io


class SingleFrequencyField:
    """单个频率场数据容器"""
    def __init__(self, df: pd.DataFrame, freq: float):
        self.dataframes = df
        self.freq = freq


class FFData:
    """场分量数据类"""
    def __init__(self, fields: List[SingleFrequencyField]):
        datas = []
        for field in fields:
            # 1. 提取坐标列以外的所有列作为数据变量
            # 此时列名已经被 Parser 标准化为 Theta, Phi
            data_vars = [c for c in field.dataframes.columns if c not in ['Theta', 'Phi']]
            
            # 2. 转换为 Xarray (自动处理多维结构)
            ds_f = field.dataframes.set_index(['Theta', 'Phi'])[data_vars].to_xarray()
            
            # 3. 扩展频率维度
            ds_f = ds_f.expand_dims({'Frequency': [field.freq]})
            datas.append(ds_f)
        
        # 4. 合并所有频点
        self.ds_multi = xr.concat(datas, dim='Frequency')
    
    @property
    def frequencies(self) -> np.ndarray:
        """返回频率数组"""
        return self.ds_multi.Frequency.values
    
    @property
    def thetas(self) -> np.ndarray:
        """返回方位角数组"""
        return self.ds_multi.Theta.values
    
    @property
    def phis(self) -> np.ndarray:
        """返回仰角数组"""
        return self.ds_multi.Phi.values
    
    @property
    def values(self) -> np.ndarray:
        """返回场分量值"""
        return self.ds_multi.to_array().values
    
    def __getitem__(self, key: str) -> xr.DataArray:
        """
        获取指定字段的场分量值
        
        Args:
            key (str): 场分量字段名称
        
        Returns:
            Fields: 场分量对象
        """
        if key in ['Frequency', 'Theta', 'Phi', r"Theta'", r"Phi'"]:
            return getattr(self, key)
        
        return self.ds_multi[key]
    
    @property
    def electric_field(self) -> xr.Dataset:
        """提取复数电场分量"""
        # 利用 xarray 的计算能力，自动对齐
        try:
            # 优先尝试标准命名
            Etheta = self.ds_multi['Re(Etheta)'] + 1j * self.ds_multi['Im(Etheta)']
            Ephi = self.ds_multi['Re(Ephi)'] + 1j * self.ds_multi['Im(Ephi)']
        except KeyError:
            # 如果解析器没清洗干净，这里做个保底，但最好是在解析器里做
            raise ValueError("找不到标准的电场分量列(Re(Etheta)等)")
            
        return xr.Dataset({'Etheta': Etheta, 'Ephi': Ephi})

    def to_cartesian(self) -> xr.Dataset:
        """球坐标转直角坐标 (返回 xarray Dataset)"""
        efield = self.electric_field
        E_theta = efield['Etheta']
        E_phi = efield['Ephi']
        
        # 使用 xarray 的广播机制，不需要手动 meshgrid
        theta = np.deg2rad(self.ds_multi.Theta)
        phi = np.deg2rad(self.ds_multi.Phi)
        
        # 假设 E_theta 和 E_phi 是 Theta, Phi 分量的复数场
        # 标准球坐标转直角坐标变换 (Ludwig 3 定义或其他定义需根据实际情况调整)
        # 这里使用标准的矢量分解
        Ex = E_theta * np.cos(theta) * np.cos(phi) - E_phi * np.sin(phi)
        Ey = E_theta * np.cos(theta) * np.sin(phi) + E_phi * np.cos(phi)
        Ez = -E_theta * np.sin(theta)
        
        return xr.Dataset({'Ex': Ex, 'Ey': Ey, 'Ez': Ez})


class FFEParser:

    @staticmethod
    def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """统一列名，去除 ' 号"""
        clean_cols = {}
        for col in df.columns:
            if "'" in col:
                clean_cols[col] = col.replace("'", "")
        return df.rename(columns=clean_cols)

    @staticmethod
    def _parse_section(section: List[str]) -> Tuple[pd.DataFrame, float]:
        """解析单个频点块"""
        freq = 0.0
        header_str = ""
        data_lines = []
        
        for line in section:
            if line.startswith('#Frequency:'):
                try:
                    freq = float(line.split(':')[-1].strip())
                except: pass
            elif line.startswith('#') and ('Theta' in line):
                header_str = line.replace('#', '').strip()
            elif not line.strip().startswith(('#', '*')) and line.strip():
                data_lines.append(line)
        
        if not header_str or not data_lines:
            raise ValueError("Invalid section data")
            
        # 使用 pandas 快速解析
        csv_io = io.StringIO(header_str + "\n" + "".join(data_lines))
        df = pd.read_csv(csv_io, sep=r'\s+')
        
        # 立即标准化
        df = FFEParser._standardize_columns(df)
        
        return df, freq

    @staticmethod
    def _parse(file_path: str) -> List[SingleFrequencyField]:
        # 读取整个文件
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            
        # 寻找分割点
        config_indices = [i for i, line in enumerate(lines) if 'Configuration Name:' in line]
        if not config_indices:
            # 也许只有一个配置，尝试直接解析
            if len(lines) > 0: config_indices = [0]
            else: return []

        results = []
        for i in range(len(config_indices)):
            start = config_indices[i]
            end = config_indices[i+1] if i+1 < len(config_indices) else len(lines)
            
            # 传入切片
            try:
                df, freq = FFEParser._parse_section(lines[start:end])
                results.append(SingleFrequencyField(df, freq))
            except ValueError:
                continue # 跳过无法解析的段（例如只有头没有数据的）
                
        return results

    @staticmethod
    @lru_cache(maxsize=32)
    def parse(*file_paths: str) -> FFData:
        all_fields = []
        for path in file_paths:
            all_fields.extend(FFEParser._parse(path))
            
        # 按频率去重并排序
        # 使用 dict 来去重 (保留最后出现的或者最先出现的，看需求)
        unique_fields = {f.freq: f for f in all_fields} 
        sorted_fields = [unique_fields[k] for k in sorted(unique_fields.keys())]
        
        return FFData(sorted_fields)

if __name__ == "__main__":
    test = ["Theta'"]
    clean_cols = {}
    for col in test:
        if "'" in col:
            clean_cols[col] = col.replace("'", "")
    print(clean_cols)
