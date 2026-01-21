import os
import time
import torch
import numpy as np
from thop import profile
import torch.autograd.profiler as profiler
from torch.cuda.amp import autocast
import psutil
import GPUtil
from collections import OrderedDict

from models.Raptor import PredictionRWKV
# from SimVP import PredictionRWKV
# from TAU import PredictionRWKV
from models.configurations import config_64, config_128, config_256, config_512, config_1024, config_1024_16g
# from CustomDatasets import UAVID_Images_Dataset_v1
from configs.CustomDatasets import UAVID_Images_Dataset_v1 as CustomDataset

class ModelProfiler:
    def __init__(self, model, config, device='cuda'):
        self.model = model
        self.config = config
        self.device = device
        self.results = OrderedDict()

    def profile_flops(self, sample_input, sample_flows):
        """计算FLOPs和参数量"""
        with torch.no_grad():
            flops, params = profile(self.model, inputs=(sample_input, sample_flows,))
            self.results['GFLOPs'] = flops / 1e9
            self.results['Parameters(M)'] = params / 1e6
            
    def profile_memory(self, sample_input, sample_flows):
        """分析内存使用情况"""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # GPU内存使用前
        gpu = GPUtil.getGPUs()[0]
        mem_before = gpu.memoryUsed
        
        with torch.no_grad():
            output = self.model(sample_input, sample_flows)
            torch.cuda.synchronize()
        
        # GPU内存使用后
        mem_after = gpu.memoryUsed
        peak_mem = torch.cuda.max_memory_allocated() / 1024**3  # GB
        
        self.results['Peak GPU Memory(GB)'] = peak_mem
        self.results['GPU Memory Usage(GB)'] = (mem_after - mem_before) / 1024**3
        
    def profile_inference_time(self, sample_input, sample_flows, num_runs=100):
        """测量推理时间"""
        self.model.eval()
        times = []
        
        with torch.no_grad():
            # 预热
            for _ in range(10):
                _ = self.model(sample_input, sample_flows)
            
            # 计时
            for _ in range(num_runs):
                start = time.time()
                _ = self.model(sample_input, sample_flows)
                torch.cuda.synchronize()
                times.append(time.time() - start)
        
        self.results['Average Inference Time(ms)'] = np.mean(times) * 1000
        self.results['Std Inference Time(ms)'] = np.std(times) * 1000
        
    def profile_throughput(self, sample_input, sample_flows, duration=10):
        """测量吞吐量"""
        self.model.eval()
        count = 0
        start_time = time.time()
        
        with torch.no_grad():
            while time.time() - start_time < duration:
                _ = self.model(sample_input, sample_flows)
                count += sample_input.size(0)
                
        throughput = count / duration
        self.results['Throughput(samples/s)'] = throughput
        
    def profile_layer_time(self, sample_input, sample_flows):
        """分析每层的时间消耗"""
        layer_times = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                if name not in layer_times:
                    layer_times[name] = 0
                layer_times[name] += time.time()
            return hook
        
        hooks = []
        for name, module in self.model.named_modules():
            hooks.append(module.register_forward_hook(hook_fn(name)))
            
        with torch.no_grad():
            _ = self.model(sample_input, sample_flows)
            
        for hook in hooks:
            hook.remove()
            
        self.results['Layer Times'] = layer_times
        
    def run_all_profiles(self, sample_input, sample_flows):
        """运行所有分析"""
        print("开始性能分析...")
        
        print("1. 分析FLOPs和参数量...")
        self.profile_flops(sample_input, sample_flows)
        
        print("2. 分析内存使用...")
        self.profile_memory(sample_input, sample_flows)
        
        print("3. 分析推理时间...")
        self.profile_inference_time(sample_input, sample_flows)
        
        print("4. 分析吞吐量...")
        self.profile_throughput(sample_input, sample_flows)
        
        print("5. 分析层级时间消耗...")
        self.profile_layer_time(sample_input, sample_flows)
        
        return self.results
    
    def print_results(self):
        """打印分析结果"""
        print("\n========== 模型性能分析报告 ==========")
        for metric, value in self.results.items():
            if metric != 'Layer Times':
                print(f"{metric}: {value:.4f}")
                
        print("\n层级时间分析:")
        if 'Layer Times' in self.results:
            sorted_layers = sorted(
                self.results['Layer Times'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            for name, time_used in sorted_layers[:10]:  
                print(f"{name}: {time_used*1000:.2f}ms")

def main():
    # 配置和初始化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = config_1024_16g
    model = PredictionRWKV(config).to(device)
    
    # 加载模型权重
    model_path = '/mnt/nas2/Outputs/gzl/Video_prediction_rwkv/UAVID/1024_16g/best_model.pth'  # 替换为你的模型路径
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # 创建数据集和获取样本
    dataset = CustomDataset(
        '/datasets_active/UAVID-images/uavid_train_sample', 
        device,
        config['desired_shape'],
        config['n_condition'],
        config['n_prediction'],
        if_flow=config['if_flow'],
        samples_per_dir=config['samples_per_dir']
    )
    
    # 获取单个样本
    sample_input, sample_flows, _, _ = dataset[0]
    sample_input = sample_input.unsqueeze(0).to(device)  # 添加batch维度
    sample_flows = sample_flows.unsqueeze(0).to(device)
    
    # 创建分析器并运行分析
    profiler = ModelProfiler(model, config, device)
    results = profiler.run_all_profiles(sample_input, sample_flows)
    profiler.print_results()
    
    # 保存结果到文件
    save_path = 'model_analysis_results.txt'
    with open(save_path, 'w') as f:
        for metric, value in results.items():
            if metric != 'Layer Times':
                f.write(f"{metric}: {value:.4f}\n")
        
        f.write("\nLayer Times:\n")
        if 'Layer Times' in results:
            sorted_layers = sorted(
                results['Layer Times'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            for name, time_used in sorted_layers:
                f.write(f"{name}: {time_used*1000:.2f}ms\n")

if __name__ == '__main__':
    main()
