# R 环境依赖

`08_ergm/01_valued_ergm_models.R` 需要以下 R 包：

- `xml2`
- `network`
- `ergm`
- `ergm.count`
- `coda`

建议使用 R 4.3 或更高版本。脚本从自身位置推导 `algorithm` 根目录，不依赖当前工作目录。由于 ERGM 完整拟合耗时较长，归档验收只进行 R 语法解析、包可用性预检、输入契约检查以及既有 `seed42` 输出哈希核验。
