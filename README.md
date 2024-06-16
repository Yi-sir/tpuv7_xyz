# tpuv7 测试代码


参考sophon-demo和sophon-stream中bmnn_utils.h头文件，使用tpuv7的接口整理的代码

```bash
./
├── CMakeLists.txt
├── compare.py              # python的简易对比脚本
├── data
│   ├── 1684
│   ├── 1684x
│   │   ├── input_fp321b    # 1684x上 fp32模型的输入
│   │   ├── input_int81b    # 1684x上 int8模型的输入
│   │   ├── output_fp321b   # 1684x上 fp32模型的输出
│   │   └── output_int81b   # 1684x上 int8模型的输出
│   └── 1690
│       ├── output_fp321b   # 1690上 fp32模型的输出
│       └── output_int81b   # 1690上 int8模型的输出
├── main.cc                 # 读入1690的模型、1684x的输入输出，使用84x的输入进行推理，将结果与84x的输出作比较并保存
├── README.md
└── tpu_utils.h             # header in bmnn_utils.h' s style
```