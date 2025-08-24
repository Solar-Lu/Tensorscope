import os
import re
import sys
import random
import string
import numpy as np
import mindspore as ms
from loguru import logger as mylogger
from datetime import datetime

# 配置日志，同时输出到控制台和文件
log_file = f"ms_fuzzer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
mylogger.remove()  # 移除默认处理器
mylogger.add(sys.stdout, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
mylogger.add(log_file, format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}", rotation="100 MB", retention="7 days")

MAX_FUZZ_ITER = 2000

def random_printable(len):
    candidate = list(string.printable)[:-7]
    res = ''.join(random.sample(candidate, len)).replace('"', '')
    return f'"{res}"'

def random_tensor(dtype):

    rand_dim = random.choice([0, 1, 2, 4, 8])
    rand_shape = [random.choice([0, 1, 2, 4, 8]) for i in range(rand_dim)]

    if 'string' in dtype:
        return random_printable(random.randint(5, 10))
    elif 'bool' in dtype:
        return random.choice(["True", "False"])
    elif 'int8' in dtype:
        lb = -128
        ub = 127
    elif 'uint8' in dtype:
        lb = 0
        ub = 255
    elif 'int16' in dtype:
        lb = -32768
        ub = 32767
    elif 'uint16' in dtype:
        lb = 0
        ub = 65536
    elif 'int32' in dtype:
        lb = -2147483648
        ub = 2147483647
    elif 'uint32' in dtype:
        lb = 0
        ub = 4294967295
    elif 'int64' in dtype:
        lb = -9223372036854775808
        ub = 9223372036854775807
    elif 'uint64' in dtype:
        lb = 0
        ub = 18446744073709551615   
    elif 'float16' in dtype:
        lb = -555
        ub = 1000
    elif 'float32' in dtype:
        lb = -6666666
        ub = 100000000
    elif 'float64' in dtype:
        lb = -6666666
        ub = 100000000  
    elif 'complex64' in dtype:
        lb = -10000
        ub = 60000 
    elif 'complex128' in dtype:
        lb = -10000
        ub = 60000
    else:
        print(f'ERROR: Does not have this type {dtype}!')
    return f"ms.Tensor(np.random.uniform({lb}, {ub}, {rand_shape}).astype(np.{dtype}))" 

def gen_fuzz_input(args, dtypes):
    res = ''
    arg_dtypes = random.choice(dtypes)
    for i in range(len(args)):
        arg_info = args[i]
        arg = arg_info['name']
        # res += f'{arg}='
        dtype = arg_dtypes[i][0]
        arg_value = random_tensor(dtype)
        res += arg_value
        res += ','
    return res

def fuzz_single(op):
    exec(f'import {op}')
    op_info = eval(f'{op}.{op}_op_info')
    mylogger.info(f'开始模糊测试操作: {op}')
    mylogger.info(f'操作信息: {op_info}')

    name = op_info['op_name']
    args = op_info['inputs']
    dtypes = op_info['dtype_format']

    mylogger.info(f'操作名称: {name}')
    mylogger.info(f'输入参数: {args}')
    mylogger.info(f'数据类型格式: {dtypes}')

    success_count = 0
    error_count = 0
    
    for round_num in range(MAX_FUZZ_ITER):
        cmd = f'{name}()('
        cmd += gen_fuzz_input(args, dtypes)
        cmd += ')'
        
        mylogger.debug(f'[第 {round_num+1} 轮] 生成的代码: {cmd}')
        
        try:
            exec(cmd)
            success_count += 1
            if round_num % 1000 == 0:  # 每1000轮输出一次进度
                mylogger.info(f'操作 {op} 第 {round_num+1} 轮测试成功，成功次数: {success_count}')
        except Exception as err:
            error_count += 1
            mylogger.warning(f'[第 {round_num+1} 轮] 错误: {err}')
            
            # 每1000个错误输出一次统计
            if error_count % 1000 == 0:
                mylogger.info(f'操作 {op} 错误统计 - 成功: {success_count}, 错误: {error_count}')
    
    # 输出操作测试总结
    mylogger.info(f'操作 {op} 测试完成')
    mylogger.info(f'总轮数: {MAX_FUZZ_ITER}')
    mylogger.info(f'成功次数: {success_count}')
    mylogger.info(f'错误次数: {error_count}')
    mylogger.info(f'成功率: {success_count/MAX_FUZZ_ITER*100:.2f}%')
    mylogger.info(f'错误率: {error_count/MAX_FUZZ_ITER*100:.2f}%')


if __name__ == '__main__':
    mylogger.info("=" * 80)
    mylogger.info("MindSpore模糊测试器启动")
    mylogger.info("=" * 80)
    
    # 动态查找MindSpore安装路径
    try:
        import mindspore
        mindspore_path = os.path.dirname(mindspore.__file__)
        import_dir = os.path.join(mindspore_path, 'ops', 'operations')
        fuzz_dir = os.path.join(mindspore_path, 'ops', '_op_impl', 'aicpu')
        
        mylogger.info(f"MindSpore路径: {mindspore_path}")
        mylogger.info(f"操作目录: {import_dir}")
        mylogger.info(f"AICPU目录: {fuzz_dir}")
        mylogger.info(f"日志文件: {log_file}")
        
        # 检查目录是否存在
        if not os.path.exists(import_dir):
            mylogger.error(f"操作目录不存在: {import_dir}")
            mylogger.error("请检查MindSpore是否正确安装")
            exit(1)
            
        if not os.path.exists(fuzz_dir):
            mylogger.error(f"AICPU目录不存在: {fuzz_dir}")
            mylogger.error("请检查MindSpore是否正确安装")
            exit(1)
            
    except ImportError:
        mylogger.error("MindSpore未安装，请先安装MindSpore")
        mylogger.error("安装命令: pip install mindspore")
        exit(1)
    except Exception as e:
        mylogger.error(f"初始化失败: {e}")
        exit(1)
    
    # 导入操作模块
    try:
        mylogger.info("开始导入操作模块...")
        for file in os.listdir(import_dir):
            x = file.replace('.py', '')
            if not x.startswith('_'):
                exec(f'from mindspore.ops.operations.{x} import *')
        mylogger.info("操作模块导入完成")
    except Exception as e:
        mylogger.error(f"导入操作模块失败: {e}")
        exit(1)
    
    # 添加AICPU路径并获取操作列表
    try:
        sys.path.append(fuzz_dir)
        ops = [x.replace('.py', '') for x in os.listdir(fuzz_dir)]
        if '__init__' in ops:
            ops.remove('__init__')
        mylogger.info(f"找到 {len(ops)} 个操作: {ops[:10]}{'...' if len(ops) > 10 else ''}")
    except Exception as e:
        mylogger.error(f"获取操作列表失败: {e}")
        exit(1)
    
    # 开始模糊测试
    mylogger.info("开始模糊测试...")
    mylogger.info(f"每个操作最大测试轮数: {MAX_FUZZ_ITER}")
    
    flag = 0
    total_ops_tested = 0
    total_errors = 0
    start_time = datetime.now()
    
    for op in ops:
        if op == 'linspace':
            flag = 1
            continue
        if flag:
            try:
                mylogger.info(f"开始测试操作: {op}")
                op_start_time = datetime.now()
                fuzz_single(op)
                op_end_time = datetime.now()
                op_duration = (op_end_time - op_start_time).total_seconds()
                mylogger.info(f"操作 {op} 测试完成，耗时: {op_duration:.2f}秒")
                total_ops_tested += 1
            except Exception as e:
                mylogger.error(f"测试操作 {op} 时出错: {e}")
                total_errors += 1
                continue
    
    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()
    
    # 输出测试摘要
    mylogger.info("=" * 80)
    mylogger.info("测试摘要")
    mylogger.info("=" * 80)
    mylogger.info(f"总测试时间: {total_duration:.2f}秒")
    mylogger.info(f"测试的操作数量: {total_ops_tested}")
    mylogger.info(f"出错的操作数量: {total_errors}")
    mylogger.info(f"日志文件位置: {os.path.abspath(log_file)}")
    mylogger.info("=" * 80)
    mylogger.info("MindSpore模糊测试器完成")
    mylogger.info("=" * 80)
