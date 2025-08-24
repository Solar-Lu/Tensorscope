#!/bin/bash

echo "=== 仅使用semgrep的约束提取脚本 ==="
echo "时间: $(date)"

# 只运行semgrep相关的约束提取
if command -v semgrep &> /dev/null; then
    echo "✓ semgrep已安装，开始提取Python约束..."
    
    # 提取TensorFlow Python约束
    if [ -d "tensorflow" ]; then
        echo "提取TensorFlow约束..."
        semgrep --config=py.Error.yaml tensorflow/tensorflow/python/ops/ -o tf.py.value_type_Error.constraints
    else
        echo "TensorFlow源码目录不存在，跳过"
    fi
    
    # 提取PyTorch Python约束
    if [ -d "pytorch" ]; then
        echo "提取PyTorch约束..."
        semgrep --config=py.Error.yaml pytorch/torch/ -o pt.py.value_type_Error.constraints
    else
        echo "PyTorch源码目录不存在，跳过"
    fi
    
    # 提取MindSpore Python约束
    if [ -d "mindspore" ]; then
        echo "提取MindSpore约束..."
        semgrep --config=py.Error.yaml mindspore/mindspore/python/mindspore/ops -o ms.py.constraints
    else
        echo "MindSpore源码目录不存在，跳过"
    fi
    
else
    echo "✗ semgrep未安装，请先安装: pip install semgrep"
fi

echo "=== 约束提取完成 ==="
