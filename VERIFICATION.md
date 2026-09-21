# 分享包核验说明

本包已核对为 34 个源码、文档和测试文件，未包含数据、权重、MSA、checkpoint、特征缓存或实验输出。

打包所在的 Windows 工作区没有可用的 Python 解释器，因此没有在本机伪造 pytest 通过结论。接收方应在具有 Python 3.11/3.12、项目依赖及 Protenix 的 Linux 环境运行：

```bash
python -m pip install -e .
python -m pip install -r requirements-extras.txt
python -m pytest tests -q
```

这些测试使用合成小文件，不会下载 ESM 权重、运行真实 jackhmmer 搜索或启动 GPU 训练。真实实验前仍应先检查输入 schema、序列分组划分、MSA 数据库来源和模型依赖版本。
