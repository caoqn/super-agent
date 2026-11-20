# 来源说明

Evoteam 方法源码由外部仓库 https://github.com/caoqn/EvoTeam 提供。本评测仓库仅保留三个数据集的适配器、实验配置和相关数据读取、评分与准备脚本；另附三个数据集的初始 agent 池及角色提示词，不分发方法核心、工具库或冻结测试池。

适配器来自原单数据集实验实现；其接口版本与公开混合评测版本不同，兼容性见 `methods/Evoteam/upstream.yaml`。GAIA、MAC-Bench、MASBench 的数据与环境需另行取得并遵循各自许可。MAC-Bench 使用上游评分器，MASBench Game worker 是上游环境的进程桥接。

原源目录没有可直接继承的 LICENSE，本次未替作者指定新许可证。
