# TYLNet

## 运行说明
在项目根目录下运行脚本时，请先进入 `TYLNet` 子目录（包含 `run.py` 的位置）：

```bash
cd TYLNet
```

核心配置位于 `configs/TCLNET_STAGE1.yaml`，常用字段：

- `attention_module`: 选择注意力/基线模块，例如 `baseline`、`baseline_skip`、`se`、`skattention`、`cbam`、`resblock+cbam`、`ca`、`eca`、`simam`。
- `use_skip_connection`: 是否为基线添加跳跃连接（仅当 `attention_module: baseline` 时生效）。
- `full_module_test_mode`: 设置为 `True` 时启用全模块测试模式；`False` 时按当前配置训练/测试。
- `early_stop_patience` 与 `early_stop_delta`: 早停机制的耐心轮数和改进阈值。

## 全模块测试模式使用方法
1. 编辑 `configs/TCLNET_STAGE1.yaml`，将 `full_module_test_mode` 改为 `True`。可根据需要调整 `name`、`checkpoints_dir` 与数据路径等其他字段。
2. 进入包含 `run.py` 的目录后，运行：

```bash
python run.py
```

脚本会依次训练/验证九种模块组合（基线、带跳连基线、SE、SKAttention、CBAM、ResBlock+CBAM、CA、ECA、SimAM），并在 `checkpoints_dir/name` 目录下生成每个变体的子目录和 `ablation_summary.txt`，记录配置、最佳 L2 误差、参数量与 GFLOPs。

## 单一配置运行
若只想运行某个模块，将 `full_module_test_mode` 设为 `False`，并在同一配置文件中直接设置 `attention_module` 与其他训练参数后运行 `python run.py` 即可。