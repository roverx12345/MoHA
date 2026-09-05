# 已校准的模型组合

这个目录只保存已完成 MoHA 校准的 planner + observer 组合，每组一个当前 JSON。历史配置通过 Git 查看；日志、轨迹、数据和密钥留在仓库外。

目录没有 JSON 时，表示尚无符合 MoHA 产物契约的完整校准结果，不放配置占位文件。`config.example.json` 是运行模板，不是已校准结果。

完成校准后，在仓库根目录执行：

```bash
PYTHONPATH=src python -m moha export --run /path/to/completed/run --output calibrated
git add calibrated
git commit -m "Update calibrated planner and observer profile"
```

导出时检查完成状态、运行身份、冻结 harness 哈希及接受历史。文件包含完整模型栈规格、预算、冻结 harness 和源运行定位信息；不复制密钥、凭据路径、视频、逐样本标签或轨迹。模型与预算需要一起使用，不能仅凭模型名称宣称结果可复现。

导出配置不会自动启用 specialist，也不会修改原运行。是否启用 OCR/ASR 以已校准的 `harness.specialists` 为准；运行模板中的 `specialists` 仅表示校准时可提出的候选。
