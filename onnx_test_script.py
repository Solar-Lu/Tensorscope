#!/usr/bin/env python3
"""
ONNX操作测试脚本
测试TensorFlow到ONNX的转换和推理一致性
"""

import os
import numpy as np
import tensorflow as tf
import onnx
import onnxruntime as ort
import tf2onnx

def test_simple_operations():
    """测试简单操作的TensorFlow到ONNX转换"""
    print("=" * 50)
    print("测试简单操作的TensorFlow到ONNX转换")
    print("=" * 50)
    
    # 测试数据
    test_data = np.array([[1.0, -2.0], [3.0, -4.0]], dtype=np.float32)
    
    # 1. 测试Abs操作
    print("\n1. 测试Abs操作:")
    try:
        # TensorFlow模型
        @tf.function
        def abs_model(x):
            return tf.math.abs(x)
        
        # 转换为ONNX
        model_proto, _ = tf2onnx.convert.from_function(
            abs_model, 
            input_signature=[tf.TensorSpec(shape=(2, 2), dtype=tf.float32)]
        )
        
        # 保存ONNX模型
        onnx.save(model_proto, "test_abs.onnx")
        
        # TensorFlow推理
        tf_result = abs_model(test_data)
        print(f"  TensorFlow结果: {tf_result.numpy()}")
        
        # ONNX推理
        ort_session = ort.InferenceSession("test_abs.onnx")
        onnx_result = ort_session.run(None, {"args_0": test_data})[0]
        print(f"  ONNX结果: {onnx_result}")
        
        # 比较结果
        if np.allclose(tf_result.numpy(), onnx_result, rtol=1e-5, atol=1e-5):
            print("  ✅ 结果一致")
        else:
            print("  ❌ 结果不一致")
            
    except Exception as e:
        print(f"  ⚠️ 测试失败: {e}")
    
    # 2. 测试Add操作
    print("\n2. 测试Add操作:")
    try:
        # TensorFlow模型
        @tf.function
        def add_model(x, y):
            return tf.math.add(x, y)
        
        # 转换为ONNX
        model_proto, _ = tf2onnx.convert.from_function(
            add_model, 
            input_signature=[
                tf.TensorSpec(shape=(2, 2), dtype=tf.float32),
                tf.TensorSpec(shape=(2, 2), dtype=tf.float32)
            ]
        )
        
        # 保存ONNX模型
        onnx.save(model_proto, "test_add.onnx")
        
        # 测试数据
        x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        y = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        
        # TensorFlow推理
        tf_result = add_model(x, y)
        print(f"  TensorFlow结果: {tf_result.numpy()}")
        
        # ONNX推理
        ort_session = ort.InferenceSession("test_add.onnx")
        onnx_result = ort_session.run(None, {"args_0": x, "args_1": y})[0]
        print(f"  ONNX结果: {onnx_result}")
        
        # 比较结果
        if np.allclose(tf_result.numpy(), onnx_result, rtol=1e-5, atol=1e-5):
            print("  ✅ 结果一致")
        else:
            print("  ❌ 结果不一致")
            
    except Exception as e:
        print(f"  ⚠️ 测试失败: {e}")

def test_complex_operations():
    """测试复杂操作的TensorFlow到ONNX转换"""
    print("\n" + "=" * 50)
    print("测试复杂操作的TensorFlow到ONNX转换")
    print("=" * 50)
    
    # 测试卷积操作
    print("\n1. 测试Conv2D操作:")
    try:
        # 创建简单的卷积模型
        @tf.function
        def conv_model(x):
            # 简单的卷积层
            conv = tf.keras.layers.Conv2D(
                filters=1, 
                kernel_size=(2, 2), 
                padding='valid',
                use_bias=False
            )
            return conv(x)
        
        # 测试数据
        input_data = np.random.randn(1, 4, 4, 1).astype(np.float32)
        
        # 转换为ONNX
        model_proto, _ = tf2onnx.convert.from_function(
            conv_model, 
            input_signature=[tf.TensorSpec(shape=(1, 4, 4, 1), dtype=tf.float32)]
        )
        
        # 保存ONNX模型
        onnx.save(model_proto, "test_conv.onnx")
        
        # TensorFlow推理
        tf_result = conv_model(input_data)
        print(f"  TensorFlow输出形状: {tf_result.shape}")
        
        # ONNX推理
        ort_session = ort.InferenceSession("test_conv.onnx")
        onnx_result = ort_session.run(None, {"args_0": input_data})[0]
        print(f"  ONNX输出形状: {onnx_result.shape}")
        
        # 比较形状
        if tf_result.shape == onnx_result.shape:
            print("  ✅ 输出形状一致")
        else:
            print("  ❌ 输出形状不一致")
            
    except Exception as e:
        print(f"  ⚠️ 测试失败: {e}")

def test_existing_models():
    """测试已存在的ONNX模型"""
    print("\n" + "=" * 50)
    print("测试已存在的ONNX模型")
    print("=" * 50)
    
    onnx_model_dir = "onnx_test/onnx_model"
    if os.path.exists(onnx_model_dir):
        onnx_files = [f for f in os.listdir(onnx_model_dir) if f.endswith('.onnx')]
        
        if onnx_files:
            print(f"发现 {len(onnx_files)} 个ONNX模型:")
            for model_file in onnx_files:
                print(f"  - {model_file}")
                
                try:
                    # 加载ONNX模型
                    model_path = os.path.join(onnx_model_dir, model_file)
                    onnx_model = onnx.load(model_path)
                    
                    # 验证模型
                    onnx.checker.check_model(onnx_model)
                    print(f"    ✅ 模型验证成功")
                    
                    # 获取模型信息
                    print(f"    输入: {[input.name for input in onnx_model.graph.input]}")
                    print(f"    输出: {[output.name for output in onnx_model.graph.output]}")
                    
                except Exception as e:
                    print(f"    ❌ 模型验证失败: {e}")
        else:
            print("未发现ONNX模型文件")
    else:
        print("ONNX模型目录不存在")

def cleanup_test_files():
    """清理测试文件"""
    test_files = ["test_abs.onnx", "test_add.onnx", "test_conv.onnx"]
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"已删除测试文件: {file}")

def main():
    """主函数"""
    print("ONNX操作测试")
    print("=" * 60)
    
    try:
        # 测试简单操作
        test_simple_operations()
        
        # 测试复杂操作
        test_complex_operations()
        
        # 测试已存在的模型
        test_existing_models()
        
        print("\n" + "=" * 60)
        print("ONNX测试完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理测试文件
        cleanup_test_files()

if __name__ == "__main__":
    main()
