import os
import argparse
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image 
from models import get_model

# 常量定义
MEAN = {
    "imagenet": [0.485, 0.456, 0.406],
    "clip": [0.48145466, 0.4578275, 0.40821073]
}

STD = {
    "imagenet": [0.229, 0.224, 0.225],
    "clip": [0.26862954, 0.26130258, 0.27577711]
}

class ForgeryDetector:
    def __init__(self, arch='res50', ckpt_path='./pretrained_weights/fc_weights.pth', device='cuda'):
        """
        初始化伪造检测器
        
        Args:
            arch: 模型架构 (如 'res50', 'CLIP:ViT-B/32' 等)
            ckpt_path: 预训练模型权重路径
            device: 运行设备 ('cuda' 或 'cpu')
        """
        self.arch = arch
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 根据架构选择统计信息
        stat_from = "imagenet" if arch.lower().startswith("imagenet") else "clip"
        
        # 数据预处理
        self.transform = transforms.Compose([
            transforms.Resize(256),  # 与验证代码保持一致
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN[stat_from], std=STD[stat_from]),
        ])
        
        # 创建模型选项对象 (模拟TrainOptions)
        self.opt = self._create_opt(arch)
        
        # 加载模型
        self.model = self._load_model(ckpt_path)
        
    def _create_opt(self, arch):
        """创建模型配置选项"""
        class Options:
            pass
        
        opt = Options()
        opt.arch = arch
        opt.head_type = "attention" if arch.startswith("CLIP:") else "fc"
        opt.shuffle = True if "shuffle" in arch.lower() else False
        opt.shuffle_times = 1
        opt.original_times = 1
        opt.patch_size = [14]
        opt.penultimate_feature = False
        opt.patch_base = False
        
        return opt
        
    def _load_model(self, ckpt_path):
        """加载预训练模型"""
        print(f"正在加载模型架构: {self.arch}")
        model = get_model(self.opt)
        
        # 加载权重
        print(f"正在加载权重: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location=self.device)
        
        # 根据模型类型加载不同的权重
        if self.opt.head_type == "fc":
            model.fc.load_state_dict(state_dict)
        elif self.opt.head_type == "attention":
            model.attention_head.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)
            
        model.eval()
        model.to(self.device)
        print("模型加载完成!")
        
        return model
    
    def preprocess_image(self, image_path):
        """
        预处理单张图片
        
        Args:
            image_path: 图片路径
            
        Returns:
            处理后的tensor
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片文件不存在: {image_path}")
            
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.transform(image).unsqueeze(0)  # 添加batch维度
            return image_tensor.to(self.device)
        except Exception as e:
            raise ValueError(f"图片预处理失败: {e}")
    
    def predict(self, image_path, return_probabilities=True):
        """
        对单张图片进行伪造检测
        
        Args:
            image_path: 图片路径
            return_probabilities: 是否返回概率值，否则返回类别
            
        Returns:
            如果return_probabilities=True: 返回 (real_prob, fake_prob)
            如果return_probabilities=False: 返回 "real" 或 "fake"
        """
        # 预处理图片
        image_tensor = self.preprocess_image(image_path)
        
        # 推理
        with torch.no_grad():
            output = self.model(image_tensor)
            
            # 处理不同的输出格式
            if output.shape[-1] == 2:
                # 二分类输出
                probs = torch.softmax(output, dim=1)[0]
                real_prob = probs[0].item()
                fake_prob = probs[1].item()
            else:
                # 单值输出 (通常用sigmoid)
                fake_prob = torch.sigmoid(output).item()
                real_prob = 1 - fake_prob
        
        if return_probabilities:
            return real_prob, fake_prob
        else:
            return "real" if real_prob > fake_prob else "fake"
    
    def batch_predict(self, image_paths, return_probabilities=True):
        """
        批量预测多张图片
        
        Args:
            image_paths: 图片路径列表
            return_probabilities: 是否返回概率值
            
        Returns:
            预测结果列表
        """
        results = []
        for image_path in image_paths:
            try:
                result = self.predict(image_path, return_probabilities)
                results.append({
                    'image_path': image_path,
                    'result': result,
                    'status': 'success'
                })
            except Exception as e:
                results.append({
                    'image_path': image_path,
                    'result': None,
                    'status': f'error: {e}'
                })
        return results

def main():
    parser = argparse.ArgumentParser(description='Face Forgery Detection Inference')
    parser.add_argument('--image_path', type=str, required=True, help='输入图片路径')
    parser.add_argument('--arch', type=str, default='res50', help='模型架构')
    parser.add_argument('--ckpt', type=str, default='./pretrained_weights/fc_weights.pth', help='模型权重路径')
    parser.add_argument('--device', type=str, default='cuda', help='运行设备 (cuda/cpu)')
    parser.add_argument('--threshold', type=float, default=0.5, help='分类阈值')
    
    args = parser.parse_args()
    
    # 初始化检测器
    detector = ForgeryDetector(
        arch=args.arch,
        ckpt_path=args.ckpt,
        device=args.device
    )
    
    # 进行预测
    try:
        real_prob, fake_prob = detector.predict(args.image_path, return_probabilities=True)
        prediction = "fake" if fake_prob > args.threshold else "real"
        
        print(f"图片路径: {args.image_path}")
        print(f"真实概率: {real_prob:.4f}")
        print(f"伪造概率: {fake_prob:.4f}")
        print(f"预测结果: {prediction}")
        print(f"置信度: {max(real_prob, fake_prob):.4f}")
        
    except Exception as e:
        print(f"预测失败: {e}")

if __name__ == '__main__':
    main()