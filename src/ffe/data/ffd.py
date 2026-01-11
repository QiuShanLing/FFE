import numpy as np
import xarray as xr

class FFData:
    """远场数据类"""
    def __init__(self, dataset: xr.Dataset) -> None:
        self.ds = dataset

    @property
    def frequencies(self) -> np.ndarray:
        return self.ds.Frequency.values
    
    @property
    def thetas(self) -> np.ndarray:
        """返回方位角数组"""
        return self.ds.Theta.values
    
    @property
    def phis(self) -> np.ndarray:
        """返回仰角数组"""
        return self.ds.Phi.values
    
    @property
    def values(self) -> np.ndarray:
        """返回场分量值"""
        return self.ds.to_array().values
    
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
        
        return self.ds[key]
    
    @property
    def electric_field(self) -> xr.Dataset:
        """提取复数电场分量"""
        # 利用 xarray 的计算能力，自动对齐
        try:
            # 优先尝试标准命名
            Etheta = self.ds['Re(Etheta)'] + 1j * self.ds['Im(Etheta)']
            Ephi = self.ds['Re(Ephi)'] + 1j * self.ds['Im(Ephi)']
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
        theta = np.deg2rad(self.ds.Theta)
        phi = np.deg2rad(self.ds.Phi)
        
        # 假设 E_theta 和 E_phi 是 Theta, Phi 分量的复数场
        # 标准球坐标转直角坐标变换 (Ludwig 3 定义或其他定义需根据实际情况调整)
        # 这里使用标准的矢量分解
        Ex = E_theta * np.cos(theta) * np.cos(phi) - E_phi * np.sin(phi)
        Ey = E_theta * np.cos(theta) * np.sin(phi) + E_phi * np.cos(phi)
        Ez = -E_theta * np.sin(theta)
        
        return xr.Dataset({'Ex': Ex, 'Ey': Ey, 'Ez': Ez})