import numpy as np
import xarray as xr

class FFEToXarray:
    def __init__(self, ffe_obj):
        """
        :param ffe_obj: C++ parser 返回的 FFEFile 对象
        """
        self.ffe = ffe_obj
        self.headers = [h.strip('"') for h in self.ffe.headers] # 去除可能存在的引号
        
        # 1. 验证数据有效性
        if not self.ffe.sections:
            raise ValueError("FFEFile contains no sections.")
        
        # 假设所有 Section 的空间点数一致（FEKO 标准）
        self.n_freq = len(self.ffe.sections)
        self.n_spatial = self.ffe.sections[0].data.shape[0]
        self.n_cols = len(self.headers)

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

        # 获取唯一的坐标值（有序）
        thetas = np.unique(theta_col)
        phis = np.unique(phi_col)
        
        n_theta = len(thetas)
        n_phi = len(phis)

        # 检查是否是规则网格
        if n_theta * n_phi != self.n_spatial:
            print(f"警告: 数据点数量 ({self.n_spatial}) 不等于 Theta({n_theta}) * Phi({n_phi})。可能不是规则网格，将退化为点云模式。")
            return None, None, None, None

        return t_idx, p_idx, thetas, phis

    def convert(self):
        # --- 步骤 1: 提取所有频率 ---
        freqs = np.array([s.frequency for s in self.ffe.sections])

        # --- 步骤 2: 构建巨型 Numpy 数组 ---
        # 预分配内存: (Frequency, Spatial_Points, Columns)
        # 直接使用 stack 可能会触发拷贝，但这是最快的方法之一
        # 如果内存极度紧张，可以使用预分配 + 循环填入，但 C++ 返回的已经是 numpy，stack 很快
        all_data = np.stack([s.data for s in self.ffe.sections], axis=0) 

        # --- 步骤 3: 解析空间网格 ---
        # 使用第一个频率的数据来确定空间结构
        t_idx, p_idx, unique_thetas, unique_phis = self._get_spatial_coords(all_data[0])

        if unique_thetas is None:
            # 非规则网格的处理逻辑（暂略，通常 FEKO 都是规则的）
            raise NotImplementedError("暂不支持非规则网格数据")

        # --- 步骤 4: 重塑 (Reshape) 数据 ---
        # 原始: (Freq, N_spatial, Cols) -> 目标: (Freq, Theta, Phi, Cols)
        # 注意：FEKO 的扫描顺序通常是 "Phi 循环在内" 或 "Theta 循环在内"
        # 我们需要判断 reshape 的顺序 ('C' or 'F')。
        # 简单判断：看前两个点的变化。如果第2个点 Phi 变了而 Theta 没变，说明 Phi 是内层循环。
        
        # 默认尝试 reshape
        reshaped_data = all_data.reshape(self.n_freq, len(unique_thetas), len(unique_phis), self.n_cols, order='F')
        
        # 验证 Reshape 是否正确：
        # 检查 reshape 后由 theta/phi 索引取出的值，是否等于坐标轴的值
        # 如果顺序反了，这里需要 swapaxes 或者以 order='F' reshape
        # 这是一个极快的验证，不会消耗太多时间
        test_theta = reshaped_data[0, :, 0, t_idx] # 取第一列 Theta
        if not np.allclose(test_theta, unique_thetas):
            # 说明存储顺序可能是 (Phi, Theta) 而不是 (Theta, Phi)
            # 调整 reshape 策略或 transpose
            reshaped_data = all_data.reshape(self.n_freq, len(unique_thetas), len(unique_phis), self.n_cols, order='F')
            # 如果还不对，可能需要先 reshape 成 (Freq, Phi, Theta, Cols) 然后 swapaxes

        # --- 步骤 5: 构建 Xarray Dataset ---
        data_vars = {}
        
        # 遍历 Header 中的每一列
        for i, col_name in enumerate(self.headers):
            # 跳过坐标列本身 (Theta, Phi)
            if i == t_idx or i == p_idx:
                continue
            
            # 提取该变量的所有数据
            # data slice: (Freq, Theta, Phi)
            var_data = reshaped_data[:, :, :, i]
            
            data_vars[col_name] = (["Frequency", "Theta", "Phi"], var_data)

        # 创建 Dataset
        ds = xr.Dataset(
            data_vars=data_vars,
            coords={
                "Frequency": freqs,
                "Theta": unique_thetas,
                "Phi": unique_phis
            },
            attrs={"description": "Parsed from FFE file"}
        )
        
        return ds