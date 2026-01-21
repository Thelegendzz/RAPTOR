base_config = {
    'device': 'cuda',
    'parallel_mode': 'ddp',  # 可选值: 'single', 'ddp', 'model_parallel'
    'backend': 'nccl',
    'master_addr': 'localhost',
    'master_port': '12355',
    'if_load_model': True,
    'if_save_interval_pkl': True,
    'interval_save_pkl': 10,
    'if_flow': 0,
    'samples_per_dir': 100,
    'n_condition': 8,
    'n_prediction': 8,
    'n_channels': 3,
    'input_channels': 3,
    'output_channels': 3,
    'dropout': 0.1,
    'n_pixels': 0,  # 将在代码中派生
    'kernel_size': 3,
    'stride': [1, 1],
    'padding': 1,

    'batch_size': 2,
    'num_workers': 2,
    'prefetch_factor':2,
    'shuffle': True,
    'seq_len': 0,  # 将在代码中派生
    't_len': 2048,

    'post_readin_dropout_p': 0.3,
    'post_embedding_dropout_p': 0.7,
    'post_encoder_dropout_p': 0.3,
    'mha_dropout_p': 0.5,
    'ffn_dropout_p': 0.3,
    
    'loss_ratio': 0.4,
    'mask_ratio': 0.0,
    'random_ratio': 1.0,
    'weight_decay': 1e-4,
    'warmup_epochs': 5,
    'lr_min': 1e-6,
    'use_ema': True,
    'ema_decay': 0.999,
    'lr_init': 1e-3,
    'n_epochs': 110,

    # TAU模块配置
    'tau': 0.1,                    # TAU正则化温度参数
    'tau_alpha': 0.1,              # TAU正则化权重
    'tau_num_heads': 8,            # TAU注意力头数
    'tau_dropout': 0.1,            # TAU dropout率

    # 损失权重配置
    'weight_l1': 0.5,        # 新增L1损失权重
    'weight_mse': 1.0,       # MSE损失权重
    'weight_ssim': 0.01,     # 结构相似性损失权重
    'weight_gdl': 0.05,      # 梯度差分损失权重
    'weight_temporal': 0.3,  # 时序一致性损失权重
    'weight_perceptual': 0.001, # 感知损失权重
    'weight_contrastive': 0.1,  # 对比损失权重
    'weight_lpips': 0.1,  # LPIPS损失的权重，根据需要调整
    'lpips_model': 'vgg',  # 可选 'alex', 'vgg'
    
    # 训练稳定性配置
    'gradient_clip_val': 1.0,    # 梯度裁剪阈值
    'accumulate_grad_batches': 2, # 梯度累积步数
    'label_smoothing': 0.1,      # 标签平滑
    
    # 优化器配置
    'optimizer': {
        'type': 'AdamW',
        'lr': 1e-4,
        'betas': (0.9, 0.999),
        'eps': 1e-8,
        'weight_decay': 0.05
    },
    
    # 学习率调度器配置
    'scheduler': {
        'warmup_epochs': 5,
        'min_lr': 1e-7,
        'decay_rate': 0.1,
        'decay_epochs': 30
    },
    
    # 模型架构配置
    'architecture': {
        'dropout': 0.1,
        'attention_dropout': 0.1,
        'path_dropout': 0.2,
        'layer_scale_init_value': 1e-6,
        'use_checkpoint': True  # 使用梯度检查点节省显存
    }
    
}
# 配置版本 v4 - 分辨率 1024x1024
config_64 = {
    **base_config,
    'lr_init': 1e-3,
    'batch_size': 16, #divided by 'samples_per_dir': 50,
    'num_workers': 8,
    'prefetch_factor':4,
    'desired_shape': [64, 64],
    'height': 64,
    'width': 64,
    'hid_S': 3,
    'n_embd': 512,
    'n_attn': 1024,
    'n_ffn': 2048,

    't_embd': 16,
    't_attn': 32,
    't_ffn': 64,
    't_ffn': 64,
    
    'N_S': 8,             # 空间编码器层数
    'N_T': 4,             # 时间编码器层数
    'spatio_kernel_enc': 3,  # 编码器卷积核大小
    'spatio_kernel_dec': 3,  # 解码器卷积核大小
    'n_layer': 16,
    'n_head': 8,
    't_head': 4,

    'optimizer': {
        'type': 'AdamW',
        'lr': 1e-3,            # 学习率调高
        'betas': (0.9, 0.999),
        'eps': 1e-8,
        'weight_decay': 1e-4  # 调整权重衰减
    },
}

config_128 = {
    **base_config,
    'batch_size': 4, #divided by 'samples_per_dir': 50,
    'num_workers': 8,
    'prefetch_factor':4,
    'desired_shape': [128, 128],
    'height': 128,
    'width': 128,
    'hid_S': 3,
    'n_embd': 512,
    'n_attn': 1024,
    'n_ffn': 2048,

    't_embd': 16,
    't_attn': 32,
    't_ffn': 64,
    't_ffn': 64,
    
    'N_S': 8,             # 空间编码器层数
    'N_T': 4,             # 时间编码器层数
    'spatio_kernel_enc': 3,  # 编码器卷积核大小
    'spatio_kernel_dec': 3,  # 解码器卷积核大小
    'n_layer': 16,
    'n_head': 8,
    't_head': 4,
}

config_256 = {
    **base_config,
    'batch_size': 16, #divided by 'samples_per_dir': 50,
    'num_workers': 8,
    'prefetch_factor':4,
    'desired_shape': [256, 256],
    'height': 256,
    'width': 256,
    'hid_S': 3,
    'n_embd': 512,
    'n_attn': 1024,
    'n_ffn': 2048,

    't_embd': 16,
    't_attn': 32,
    't_ffn': 64,
    't_ffn': 64,
    
    'N_S': 8,             # 空间编码器层数
    'N_T': 4,             # 时间编码器层数
    'spatio_kernel_enc': 3,  # 编码器卷积核大小
    'spatio_kernel_dec': 3,  # 解码器卷积核大小
    'n_layer': 16,
    'n_head': 8,
    't_head': 4,
}

config_512 = {
    **base_config,
    'batch_size': 2, #divided by 'samples_per_dir': 50,
    'num_workers': 8,
    'prefetch_factor':4,
    'desired_shape': [512, 512],
    'height': 512,
    'width': 512,
    'hid_S': 64,
    'n_embd': 2048,
    'n_attn': 4096,
    'n_ffn': 8192,

    't_embd': 32,
    't_attn': 64,
    't_ffn': 128,
    
    'N_S': 6,             # 空间编码器层数
    'N_T': 6,             # 时间编码器层数
    'spatio_kernel_enc': 3,  # 编码器卷积核大小
    'spatio_kernel_dec': 3,  # 解码器卷积核大小
    'n_layer': 16,
    'n_head': 8,
    't_head': 4,
    'lr_init': 1e-4,
}

config_1024 = {
    **base_config,
    'batch_size': 2, #divided by 'samples_per_dir': 50,
    'num_workers': 8,
    'prefetch_factor':4,
    'desired_shape': [1024, 1024],
    'height': 1024,
    'width': 1024,
    'hid_S': 3,
    'n_embd': 1024,
    'n_attn': 2048,
    'n_ffn': 4096,

    't_embd': 16,
    't_attn': 32,
    't_ffn': 64,
    
    'N_S': 8,             # 空间编码器层数
    'N_T': 4,             # 时间编码器层数
    'spatio_kernel_enc': 3,  # 编码器卷积核大小
    'spatio_kernel_dec': 3,  # 解码器卷积核大小
    'n_layer': 16,
    'n_head': 8,
    't_head': 4,
}

config_1024_16g = {
    **base_config,
    'batch_size': 1,  # 减少batch_size从2到1
    'num_workers': 8,  # 减少worker数量
    'prefetch_factor': 4,  # 减少预取因子
    'desired_shape': [1024, 1024],
    'height': 1024,
    'width': 1024,
    'hid_S': 3,
    'n_embd': 512,  # 减少嵌入维度从1024到512
    'n_attn': 1024,  # 减少注意力维度从2048到1024
    'n_ffn': 2048,  # 减少前馈网络维度从4096到2048

    't_embd': 16,  # 减少时间嵌入维度从16到8
    't_attn': 32,  # 减少时间注意力维度从32到16
    't_ffn': 64,  # 减少时间前馈网络维度从64到32
    
    'N_S': 6,  # 减少空间编码器层数从8到4
    'N_T': 4,  # 减少时间编码器层数从4到2
    'spatio_kernel_enc': 3,
    'spatio_kernel_dec': 3,
    'n_layer': 16,  # 减少总层数从16到8
    'n_head': 8,  # 减少注意力头数从8到4
    't_head': 4,  # 减少时间注意力头数从4到2
    
    # 启用梯度检查点以节省显存
    'use_checkpoint': True,
    'gradient_clip_val': 0.5,
    'accumulate_grad_batches': 4,  # 增加梯度累积以补偿batch_size减少
    
    # 调整学习率
    'lr_init': 5e-5,  # 降低学习率
    'lr_min': 5e-7,
}