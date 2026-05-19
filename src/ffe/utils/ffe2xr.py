import numpy as np
import xarray as xr


class FFEToXarray:
    def __init__(self, ffe_obj=None, *, headers=None, frequencies=None, data=None):
        """
        :param ffe_obj: C++ parser 返回的 FFEFile 对象，或 parse_ffe_array 的结果
        """
        if data is None and isinstance(ffe_obj, tuple) and len(ffe_obj) == 3:
            headers, frequencies, data = ffe_obj

        if data is None:
            if ffe_obj is None or not ffe_obj.sections:
                raise ValueError("FFEFile contains no sections.")
            headers = ffe_obj.headers
            frequencies = np.array([s.frequency for s in ffe_obj.sections])
            data = np.stack([s.data for s in ffe_obj.sections], axis=0)

        self.headers = [h.strip('"').replace("'", "") for h in headers]
        self.frequencies = np.asarray(frequencies)
        self.data = np.asarray(data)

        if self.data.ndim != 3:
            raise ValueError("FFE data must have shape (Frequency, SpatialPoint, Column).")

        self.n_freq, self.n_spatial, self.n_cols = self.data.shape
        if self.n_cols != len(self.headers):
            raise ValueError("FFE header count does not match data column count.")

    @classmethod
    def from_file(cls, path):
        from ..parser import parse_ffe_array

        headers, frequencies, data = parse_ffe_array(path)
        return cls(headers=headers, frequencies=frequencies, data=data)

    def _get_spatial_coords(self, data_sample):
        """
        从第一帧数据中解析 Theta 和 Phi 的网格结构。
        data_sample: shape (N_spatial, N_cols)
        """
        # 找到 Theta 和 Phi 在列中的索引
        # 注意：FEKO headers 通常是 "Theta", "Phi" (区分大小写，根据实际情况调整)
        try:
            t_idx = self.headers.index("Theta")
            p_idx = self.headers.index("Phi")
        except ValueError:
            # 容错：如果找不到，尝试全小写匹配或其他备选方案
            lower_headers = [h.lower() for h in self.headers]
            t_idx = lower_headers.index("theta")
            p_idx = lower_headers.index("phi")

        # 提取坐标列
        theta_col = data_sample[:, t_idx]
        phi_col = data_sample[:, p_idx]

        thetas = np.unique(theta_col)
        phis = np.unique(phi_col)
        
        n_theta = len(thetas)
        n_phi = len(phis)

        # 检查是否是规则网格
        if n_theta * n_phi != self.n_spatial:
            return None, None, None, None

        return t_idx, p_idx, thetas, phis

    def _reshape_to_grid(self, t_idx, p_idx, thetas, phis):
        n_theta = len(thetas)
        n_phi = len(phis)
        sample = self.data[0]

        theta_outer = sample[:, t_idx].reshape(n_theta, n_phi)
        phi_inner = sample[:, p_idx].reshape(n_theta, n_phi)
        if (
            np.allclose(theta_outer, thetas[:, None])
            and np.allclose(phi_inner, phis[None, :])
        ):
            return self.data.reshape(self.n_freq, n_theta, n_phi, self.n_cols)

        phi_outer_theta_inner = sample[:, t_idx].reshape(n_phi, n_theta).T
        theta_inner_phi_outer = sample[:, p_idx].reshape(n_phi, n_theta).T
        if (
            np.allclose(phi_outer_theta_inner, thetas[:, None])
            and np.allclose(theta_inner_phi_outer, phis[None, :])
        ):
            return self.data.reshape(self.n_freq, n_phi, n_theta, self.n_cols).swapaxes(1, 2)

        return self._reshape_by_coordinate_index(t_idx, p_idx, thetas, phis)

    def _reshape_by_coordinate_index(self, t_idx, p_idx, thetas, phis):
        theta_index = {value: idx for idx, value in enumerate(thetas)}
        phi_index = {value: idx for idx, value in enumerate(phis)}
        result = np.empty((self.n_freq, len(thetas), len(phis), self.n_cols), dtype=self.data.dtype)

        for row_idx, row in enumerate(self.data[0]):
            theta_pos = theta_index[row[t_idx]]
            phi_pos = phi_index[row[p_idx]]
            result[:, theta_pos, phi_pos, :] = self.data[:, row_idx, :]

        return result

    def convert(self):
        t_idx, p_idx, unique_thetas, unique_phis = self._get_spatial_coords(self.data[0])

        if unique_thetas is None:
            raise NotImplementedError("暂不支持非规则网格数据")

        reshaped_data = self._reshape_to_grid(t_idx, p_idx, unique_thetas, unique_phis)

        data_vars = {}
        for i, col_name in enumerate(self.headers):
            if i == t_idx or i == p_idx:
                continue

            data_vars[col_name] = (["Frequency", "Theta", "Phi"], reshaped_data[:, :, :, i])

        return xr.Dataset(
            data_vars=data_vars,
            coords={
                "Frequency": self.frequencies,
                "Theta": unique_thetas,
                "Phi": unique_phis
            },
            attrs={"description": "Parsed from FFE file"}
        )
