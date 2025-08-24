#!/usr/bin/env python3
"""
高级的跨框架一致性测试脚本
测试更多TensorFlow、PyTorch和ONNX之间的操作一致性
"""

import os
import numpy as np
import tensorflow as tf
import torch
import onnx
import onnxruntime as ort
import json
import random

def test_math_operations():
    """测试数学操作的一致性"""
    print("=" * 50)
    print("测试数学操作的一致性")
    print("=" * 50)
    
    # 测试数据
    data = np.array([[1.0, -2.0], [3.0, -4.0]], dtype=np.float32)
    
    operations = [
        ('abs', 'tf.math.abs', 'torch.abs'),
        ('square', 'tf.math.square', 'torch.square'),
        ('sqrt', 'tf.math.sqrt', 'torch.sqrt'),
        ('exp', 'tf.math.exp', 'torch.exp'),
        ('log', 'tf.math.log', 'torch.log'),
    ]
    
    for op_name, tf_op, torch_op in operations:
        print(f"\n测试 {op_name} 操作:")
        
        try:
            # TensorFlow
            tf_tensor = tf.constant(data)
            tf_result = eval(tf_op)(tf_tensor)
            tf_np = tf_result.numpy()
            
            # PyTorch
            torch_tensor = torch.tensor(data)
            torch_result = eval(torch_op)(torch_tensor)
            torch_np = torch_result.numpy()
            
            # 比较结果
            if np.allclose(tf_np, torch_np, rtol=1e-5, atol=1e-5):
                print(f"  ✅ {op_name} 结果一致")
            else:
                print(f"  ❌ {op_name} 结果不一致")
                print(f"     差异: {np.max(np.abs(tf_np - torch_np))}")
                
        except Exception as e:
            print(f"  ⚠️ {op_name} 测试失败: {e}")

def test_binary_operations():
    """测试二元操作的一致性"""
    print("\n" + "=" * 50)
    print("测试二元操作的一致性")
    print("=" * 50)
    
    # 测试数据
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    
    operations = [
        ('add', 'tf.math.add', 'torch.add'),
        ('multiply', 'tf.math.multiply', 'torch.mul'),
        ('maximum', 'tf.math.maximum', 'torch.maximum'),
        ('minimum', 'tf.math.minimum', 'torch.minimum'),
    ]
    
    for op_name, tf_op, torch_op in operations:
        print(f"\n测试 {op_name} 操作:")
        
        try:
            # TensorFlow
            tf_a = tf.constant(a)
            tf_b = tf.constant(b)
            tf_result = eval(tf_op)(tf_a, tf_b)
            tf_np = tf_result.numpy()
            
            # PyTorch
            torch_a = torch.tensor(a)
            torch_b = torch.tensor(b)
            torch_result = eval(torch_op)(torch_a, torch_b)
            torch_np = torch_result.numpy()
            
            # 比较结果
            if np.allclose(tf_np, torch_np, rtol=1e-5, atol=1e-5):
                print(f"  ✅ {op_name} 结果一致")
            else:
                print(f"  ❌ {op_name} 结果不一致")
                print(f"     差异: {np.max(np.abs(tf_np - torch_np))}")
                
        except Exception as e:
            print(f"  ⚠️ {op_name} 测试失败: {e}")

def test_reduction_operations():
    """测试归约操作的一致性"""
    print("\n" + "=" * 50)
    print("测试归约操作的一致性")
    print("=" * 50)
    
    # 测试数据
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    
    operations = [
        ('reduce_sum', 'tf.math.reduce_sum', 'torch.sum'),
        ('reduce_mean', 'tf.math.reduce_mean', 'torch.mean'),
        ('reduce_max', 'tf.math.reduce_max', 'torch.max'),
        ('reduce_min', 'tf.math.reduce_min', 'torch.min'),
    ]
    
    for op_name, tf_op, torch_op in operations:
        print(f"\n测试 {op_name} 操作:")
        
        try:
            # TensorFlow
            tf_tensor = tf.constant(data)
            tf_result = eval(tf_op)(tf_tensor)
            tf_np = tf_result.numpy()
            
            # PyTorch
            torch_tensor = torch.tensor(data)
            torch_result = eval(torch_op)(torch_tensor)
            torch_np = torch_result.numpy()
            
            # 比较结果
            if np.allclose(tf_np, torch_np, rtol=1e-5, atol=1e-5):
                print(f"  ✅ {op_name} 结果一致")
            else:
                print(f"  ❌ {op_name} 结果不一致")
                print(f"     差异: {np.max(np.abs(tf_np - torch_np))}")
                
        except Exception as e:
            print(f"  ⚠️ {op_name} 测试失败: {e}")

def test_onnx_advanced():
    """测试高级ONNX操作"""
    print("\n" + "=" * 50)
    print("测试高级ONNX操作")
    print("=" * 50)
    
    import onnx.helper as helper
    
    # 创建更复杂的ONNX模型
    X = helper.make_tensor_value_info('X', onnx.TensorProto.FLOAT, [2, 3])
    Y = helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT, [2, 3])
    
    # 添加操作
    add_node = helper.make_node('Add', inputs=['X', 'Y'], outputs=['Z'])
    
    # 激活函数
    relu_node = helper.make_node('Relu', inputs=['Z'], outputs=['A'])
    
    # 创建图
    graph = helper.make_graph(
        [add_node, relu_node],
        'test-add-relu',
        [X, Y],
        [helper.make_tensor_value_info('A', onnx.TensorProto.FLOAT, [2, 3])],
    )
    
    # 创建模型
    model = helper.make_model(graph)
    
    # 验证模型
    onnx.checker.check_model(model)
    print("✅ 复杂ONNX模型创建和验证成功")
    
    # 测试推理
    data_x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    data_y = np.array([[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]], dtype=np.float32)
    
    # 使用ONNX Runtime推理
    session = ort.InferenceSession(model.SerializeToString())
    result = session.run(['A'], {'X': data_x, 'Y': data_y})
    
    print(f"ONNX Add+ReLU结果:\n{result[0]}")
    
    # 与NumPy比较
    numpy_result = np.maximum(0, data_x + data_y)
    if np.allclose(result[0], numpy_result, rtol=1e-5, atol=1e-5):
        print("✅ ONNX结果与NumPy一致")
    else:
        print("❌ ONNX结果与NumPy不一致")
        print(f"差异: {np.max(np.abs(result[0] - numpy_result))}")

def test_random_operations():
    """随机测试一些操作"""
    print("\n" + "=" * 50)
    print("随机测试一些操作")
    print("=" * 50)
    
    # 读取API映射
    with open('src/counterpart_api/onnx2tf.json', 'r') as f:
        onnx2tf_map = json.load(f)
    
    # 随机选择5个操作进行测试
    test_ops = random.sample(list(onnx2tf_map.keys()), 5)
    
    for op_name in test_ops:
        mapping = onnx2tf_map[op_name]
        print(f"\n操作: {op_name}")
        print(f"  TensorFlow对应: {mapping['ops']}")
        print(f"  伪操作: {mapping['pseudo_op']}")
        print(f"  硬模板: {mapping['hard_template']}")

def main():
    """主函数"""
    print("TensorScope 高级跨框架一致性测试")
    print("=" * 60)
    
    try:
        # 测试数学操作
        test_math_operations()
        
        # 测试二元操作
        test_binary_operations()
        
        # 测试归约操作
        test_reduction_operations()
        
        # 测试高级ONNX操作
        test_onnx_advanced()
        
        # 随机测试操作
        test_random_operations()
        
        print("\n" + "=" * 60)
        print("所有高级测试完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
