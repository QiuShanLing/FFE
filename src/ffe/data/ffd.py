from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from typing import Iterable, Union

import numpy as np
import xarray as xr

PathInput = Union[str, PathLike[str]]


@dataclass(slots=True)
class FFData:
    """远场数据类"""
    ds: xr.Dataset

    def __post_init__(self) -> None:
        if not isinstance(self.ds, xr.Dataset):
            raise TypeError(f"FFData 只接受 xarray.Dataset，当前类型为: {type(self.ds).__name__}")

    @classmethod
    def from_path(cls, path: PathInput | Iterable[PathInput]) -> FFData:
        """从一个或多个 FFE 路径读取数据并构造 FFData。"""
        from ..parser import parse_ffe_dataset, parse_ffe_datasets

        if isinstance(path, (str, PathLike)):
            return cls(parse_ffe_dataset(path))

        return cls(parse_ffe_datasets(path))

    @property
    def frequencies(self) -> np.ndarray:
        return self.ds.Frequency.values
    
    @property
    def f(self) -> np.ndarray:
        """frequency别名"""
        return self.frequencies

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
        if key in self.ds.coords:
            return self.ds.coords[key]
        
        return self.ds[key]

    def _electric_field_components(self) -> tuple[xr.DataArray, xr.DataArray]:
        """提取复数球坐标电场分量。"""
        try:
            etheta = self.ds["Re(Etheta)"] + 1j * self.ds["Im(Etheta)"]
            ephi = self.ds["Re(Ephi)"] + 1j * self.ds["Im(Ephi)"]
        except KeyError:
            raise ValueError("文件解析错误! 找不到标准的电场分量列(Re(Etheta)等)")

        return etheta, ephi

    def ff(self, theta: float, phi: float, coord: str = "sph") -> np.ndarray:
        """
        返回某点的电场，theta 和 phi 的单位均为 deg。

        `coord='sph'` 返回 `[Etheta, Ephi]`；
        `coord='cart'` 返回 `[Ex, Ey, Ez]`。
        """
        etheta, ephi = self._electric_field_components()
        specified_et = np.asarray(etheta.interp(Theta=theta, Phi=phi).values)
        specified_ep = np.asarray(ephi.interp(Theta=theta, Phi=phi).values)
        e_spherical = np.stack([specified_et, specified_ep], axis=0)

        if coord == "sph":
            return e_spherical
        if coord == "cart":
            return self._spherical_to_cartesian_at_angle(e_spherical, theta, phi)

        raise TypeError(f"不支持的坐标系类型: {coord}")

    def _spherical_to_cartesian_at_angle(self, etp: np.ndarray, theta: float, phi: float) -> np.ndarray:
        """将某个朝向的电场从球坐标转换为笛卡尔坐标。"""
        e_theta = etp[0]
        e_phi = etp[1]

        theta_rad = np.deg2rad(theta)
        phi_rad = np.deg2rad(phi)

        ex = e_theta * np.cos(theta_rad) * np.cos(phi_rad) - e_phi * np.sin(phi_rad)
        ey = e_theta * np.cos(theta_rad) * np.sin(phi_rad) + e_phi * np.cos(phi_rad)
        ez = -e_theta * np.sin(theta_rad)
        return np.stack([ex, ey, ez], axis=0)

    @property
    def ef(self) -> np.ndarray:
        """返回所有球坐标系电场分量。"""
        etheta, ephi = self._electric_field_components()
        return np.stack([etheta.values, ephi.values], axis=0)

    @property
    def exyz(self) -> np.ndarray:
        """返回所有笛卡尔坐标系电场分量。"""
        return self._spherical_to_cartesian()

    @property
    def electric_field(self) -> xr.Dataset:
        """提取复数电场分量，保留 xarray 形式。"""
        etheta, ephi = self._electric_field_components()
        return xr.Dataset({"Etheta": etheta, "Ephi": ephi})

    def to_cartesian(self) -> xr.Dataset:
        """球坐标转直角坐标，返回 xarray Dataset。"""
        etheta, ephi = self._electric_field_components()

        theta = np.deg2rad(self.ds.Theta)
        phi = np.deg2rad(self.ds.Phi)

        ex = etheta * np.cos(theta) * np.cos(phi) - ephi * np.sin(phi)
        ey = etheta * np.cos(theta) * np.sin(phi) + ephi * np.cos(phi)
        ez = -etheta * np.sin(theta)

        return xr.Dataset({"Ex": ex, "Ey": ey, "Ez": ez})

    def _spherical_to_cartesian(self) -> np.ndarray:
        """将所有球坐标电场分量转换为笛卡尔坐标。"""
        cartesian = self.to_cartesian()
        return np.stack(
            [cartesian["Ex"].values, cartesian["Ey"].values, cartesian["Ez"].values],
            axis=0,
        )
