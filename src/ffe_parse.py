import numpy as np
import pandas as pd
from typing import List
import xarray as xr
from functools import lru_cache

class SingleFrequencyField:
    """单个频率场数据容器"""
    def __init__(self, df: pd.DataFrame, freq: float, keys: List):
        '''
        Args:
            df (pd.DataFrame): 单个频率场数据
            freq (float): 频率
        '''
        self.dataframes = df
        self.freq = freq
        self.keys = keys


class FFData:
    """场分量数据类，用于存储和处理场分量数据"""
    def __init__(self, fields: List[SingleFrequencyField]):
        """
        Args:
            values (np.ndarray): 场分量值
            thetas (np.ndarray): 方位角数组
            phis (np.ndarray): 仰角数组
        """
        keys = fields[0].keys
        keys = [key for key in keys if key not in ['Theta', 'Phi']]  # 去除方位角和仰角字段
        datas = []
        for field in fields:
            ds_f = field.dataframes.set_index(['Theta', 'Phi'])[keys].to_xarray()
            ds_f = ds_f.expand_dims({'Frequency': [field.freq]})
            datas.append(ds_f)
        
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
        if key in ['Frequency', 'Theta', 'Phi']:
            return getattr(self, key)
        
        return self.ds_multi[key]
    
    @property
    def electric_field(self) -> xr.Dataset:
        '''
        提取电厂场分量
        
        Returns:
            Fields: DataArray形式的电场分量对象
        '''
        Etheta = self['Re(Etheta)'] + 1j * self['Im(Etheta)']
        Ephi = self['Re(Ephi)'] + 1j * self['Im(Ephi)']
        return xr.Dataset({'Etheta': Etheta, 'Ephi': Ephi})


    def to_cartesian(self):
        """
        将球坐标场分量转换为直角坐标系
        
        Returns:
            Fields: 包含直角坐标系分量 (ex, ey, ez) 的新Fields实例
        """
        # 转换为弧度
        theta = np.deg2rad(self.thetas)
        phi = np.deg2rad(self.phis)

        # 创建二维角度网格
        theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')

        # 检查形状是否匹配
        if self.values.shape != (len(self.thetas), len(self.phis)):
            raise ValueError(f"Shape mismatch: values {self.values.shape} must match grid {len(self.thetas), len(self.phis)}")

        # 计算直角坐标分量
        ex = self.values * np.cos(theta_grid) * np.cos(phi_grid)
        ey = self.values * np.cos(theta_grid) * np.sin(phi_grid)
        ez = -self.values * np.sin(theta_grid)
        
        # 将三个分量堆叠成三维数组
        cartesian_values = np.stack([ex, ey, ez], axis=-1)
        
        # 返回新的Fields实例
        return FFData(cartesian_values, self.thetas, self.phis)

class SpatialConfig:
    """
    场设置类，用于存储场设置信息
    如频率、方位角、仰角、场分量名称等。
    """
    def __init__(self):
        pass


class FFEParser:
    """
    FFE文件解析器，用于处理FEKO生成的.ffe远场数据文件。
    """
    
    def __init__(self, file_path: str):
        """初始化解析器并加载文件
        
        Args:
            file_path (str): .ffe文件路径
        """
        self.file_path = file_path
        self.keys = []
        self.field_contexts = []
        self.single_freq_patterns = None
        
    @staticmethod
    def _extract_all_lines_to_list(file_path) -> List[str]:
        """
        将场数据分解为单频点数据, 以列表形式返回
        """
        print(f"Reading {file_path} from disk...")
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        config_lines = []
        for i, line in enumerate(lines):
            '''
            记录所有Configuration Name行的索引，用于提取各频点数据
            '''
            if 'Configuration Name:' in line:
                config_lines.append(i)
                
        if not config_lines:
            '''
            通过config_lines判断是否存在频点数据，如果不存在，则抛出异常
            '''
            raise ValueError("未找到Configuration行, 请确认文件格式是否争取。")
            
        list_part_data = []
        for i in range(len(config_lines)):
            '''
            提取所有频点的场数据，记录在config_sections[list]列表中
            '''
            start = config_lines[i]
            end = config_lines[i+1] if i+1 < len(config_lines) else len(lines)
            list_part_data.append(lines[start:end])

        field_contexts =  list_part_data
        return field_contexts
    
    @staticmethod
    def _parse_section(section: List[str]) -> pd.DataFrame:
        """解析单个频点的数据, 将数据转换为Panda.DataFrame类型"""
        data_values = []
        keys = []
        try:
            for line in section:
                if line.startswith('#Frequency:'):
                    freq = float(line.split(':')[-1].strip())
                if line.startswith('# '):
                    keys = FFEParser._extract_keys(line)
                elif line.startswith(' '):
                    try:
                        # 增强数据转换逻辑，支持科学计数法
                        values = list(map(float, line.strip().split()))
                        data_values.append(values)
                    except ValueError as ve:
                        print(f"数据转换错误: {ve} 行内容: {line.strip()}")
                        continue

            # 验证数据完整性（根据原始transform_config_data_to_table函数逻辑）
            if not data_values:
                raise ValueError("未找到有效数据，请检查配置段格式")
            if not keys:
                raise ValueError("未找到列标题，请确认文件包含'# Theta Phi...'格式的标题行")

            # 创建DataFrame并验证数据一致性
            df = pd.DataFrame(data_values, columns=keys)
            if df.empty:
                raise ValueError("生成的DataFrame为空，请检查数据格式")
                
            # 添加原始函数中的数据类型转换
            numeric_cols = df.columns.difference(['Configuration Name'])
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
            
            # print(f"成功解析单频点数据，维度: {df.shape}")
            return df, freq, keys

        except Exception as e:
            raise ValueError(f"解析失败: {str(e)}")
    
    @staticmethod
    def _extract_keys(line: str) -> List[str]:
        try:
            keys = [element.strip('"') for element in line.strip().split(' ') if element.strip()!='' and element.strip()!='#']
            return keys
        except Exception as e:
            raise ValueError(f"提取键名时发生错误: {e}")

    @staticmethod
    def _parse(file_path: str) -> List[SingleFrequencyField]:
        """解析所有配置段为Config对象列表"""
        print(f"开始解析文件: {file_path}")
        field_contexts = FFEParser._extract_all_lines_to_list(file_path)
        single_freq_patterns = [SingleFrequencyField(*FFEParser._parse_section(field_context)) for field_context in field_contexts]
        print(f"成功解析文件: {file_path}")
        return single_freq_patterns
    
    @staticmethod
    @lru_cache(maxsize=100)  #最大缓存结果100个
    def parse(*file_paths: str) -> FFData:
        """
        解析.ffe文件, 返回FFData对象.

        :param *file_paths 远场文件路径(可以输入多个)
        """
        single_freq_patterns = [sfp for file_path in file_paths for sfp in FFEParser._parse(file_path)]
        # 多个文件导入时会有重复频点，这里只保留第一次出现的数据(与文件路径顺序相同)
        # 通过字典推导式用键的唯一性去重相同频率的方向图数据(这里用了反向遍历为了保留首次出现的元素)
        single_freq_patterns = list({sfp.freq: sfp for sfp in reversed(single_freq_patterns)}.values())[::-1]
        return FFData(single_freq_patterns)


if __name__ == '__main__':
    pass
