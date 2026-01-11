# FFE Parser - 远场数据文件处理工具
[![Version](https://img.shields.io/badge/version-1.0.0-green.svg)]()
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)]()
[![Dependencies](https://img.shields.io/badge/dependencies-pandas|xarray|numpy-orange.svg)]()

FFE Parser是一个基于Python与C++的高效工具，专门用于解析和处理远场数据文件(.ffe)。

## 功能特性
- 快速解析球坐标系的.ffe文件
- 将数据转换为封装的xarray的FFData格式
- 轻量级且易于集成到现有工作流中

## 安装指南

### 前提条件
- Python 3.11
  
### 安装方式
1. **通过Git安装**:
```bash
git clone https://github.com/QiuShanLing/FFE.git
cd FFE
uv sync && uv pip install -e .
```
## TODO

- [ ] 多频段数据合并
- [ ] 兼容warp的文件

## 许可证
本项目采用[MIT许可证](LICENSE)。

## 支持与联系
如有任何问题或建议，请通过以下方式联系我们：
- 提交[Issue](https://github.com/QiuSHanLing/FFE/issues)
- 邮箱: wzwymail@163.com
