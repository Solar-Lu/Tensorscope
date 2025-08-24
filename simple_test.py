#!/usr/bin/env python3
"""
简单的跨框架一致性测试脚本
测试TensorFlow、PyTorch和ONNX之间的基本操作一致性
"""

import os
import numpy as np
import tensorflow as tf
import torch
import onnx
import onnxruntime as ort

def test_basic_operations():
    """测试基本数学操作的一致性"""
    print("=" * 50)
    print("测试基本数学操作的一致性")
    print("=" * 50)
    
    # 创建测试数据
    data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    
    # TensorFlow
    tf_tensor = tf.constant(data)
    tf_result = tf.math.abs(tf_tensor)
    print(f"TensorFlow abs结果:\n{tf_result.numpy()}")
    
    # PyTorch
    torch_tensor = torch.tensor(data)
    torch_result = torch.abs(torch_tensor)
    print(f"PyTorch abs结果:\n{torch_result.numpy()}")
    
    # 比较结果
    tf_np = tf_result.numpy()
    torch_np = torch_result.numpy()
    
    if np.allclose(tf_np, torch_np):
        print("✅ TensorFlow和PyTorch结果一致")
    else:
        print("❌ TensorFlow和PyTorch结果不一致")
        print(f"差异: {np.max(np.abs(tf_np - torch_np))}")

def test_onnx_operations():
    """测试ONNX操作"""
    print("\n" + "=" * 50)
    print("测试ONNX操作")
    print("=" * 50)
    
    # 创建一个简单的ONNX模型
    import onnx.helper as helper
    
    # 定义输入
    X = helper.make_tensor_value_info('X', onnx.TensorProto.FLOAT, [2, 2])
    Y = helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT, [2, 2])
    
    # 定义输出
    Z = helper.make_tensor_value_info('Z', onnx.TensorProto.FLOAT, [2, 2])
    
    # 创建Add节点
    node = helper.make_node(
        'Add',
        inputs=['X', 'Y'],
        outputs=['Z'],
    )
    
    # 创建图
    graph = helper.make_graph(
        [node],
        'test-add',
        [X, Y],
        [Z],
    )
    
    # 创建模型
    model = helper.make_model(graph)
    
    # 验证模型
    onnx.checker.check_model(model)
    print("✅ ONNX模型创建和验证成功")
    
    # 测试推理
    data_x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    data_y = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    
    # 使用ONNX Runtime推理
    session = ort.InferenceSession(model.SerializeToString())
    result = session.run(['Z'], {'X': data_x, 'Y': data_y})
    
    print(f"ONNX Add结果:\n{result[0]}")
    
    # 与NumPy比较
    numpy_result = data_x + data_y
    if np.allclose(result[0], numpy_result):
        print("✅ ONNX结果与NumPy一致")
    else:
        print("❌ ONNX结果与NumPy不一致")

def test_api_mapping():
    """测试API映射文件"""
    print("\n" + "=" * 50)
    print("测试API映射文件")
    print("=" * 50)
    
    import json
    
    # 读取ONNX到TensorFlow的映射
    with open('src/counterpart_api/onnx2tf.json', 'r') as f:
        onnx2tf_map = json.load(f)
    
    print(f"ONNX到TensorFlow映射包含 {len(onnx2tf_map)} 个操作")
    
    # 显示前几个映射
    print("\n前5个映射:")
    for i, (op_name, mapping) in enumerate(list(onnx2tf_map.items())[:5]):
        print(f"  {op_name} -> {mapping['ops']}")
    
    # 测试一个具体的映射
    if 'Add' in onnx2tf_map:
        add_mapping = onnx2tf_map['Add']
        print(f"\nAdd操作映射:")
        print(f"  TensorFlow操作: {add_mapping['ops']}")
        print(f"  伪操作: {add_mapping['pseudo_op']}")
        print(f"  硬模板: {add_mapping['hard_template']}")

def main():
    """主函数"""
    print("TensorScope 跨框架一致性测试")
    print("=" * 60)
    
    try:
        # 测试基本操作
        test_basic_operations()
        
        # 测试ONNX操作
        test_onnx_operations()
        
        # 测试API映射
        test_api_mapping()
        
        print("\n" + "=" * 60)
        print("所有测试完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
