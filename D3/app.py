import gradio as gr
from PIL import Image
import numpy as np
import os
import torch
import torchvision.transforms as transforms
from models import get_model

# os.environ["http_proxt"] = ""
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
    def __init__(self, arch='CLIP:ViT-L/14', ckpt_path='path/to/save/checkpoints/train_d3/model_epoch_best.pth', device='cuda'):
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
        opt.shuffle = True 
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
    
    def preprocess_image(self, image:Image.Image):
        """
        预处理单张图片
        
        Args:
            image_path: 图片路径
            
        Returns:
            处理后的tensor
        """

            
        try:
            image_tensor = self.transform(image).unsqueeze(0)  # 添加batch维度
            return image_tensor.to(self.device)
        except Exception as e:
            raise ValueError(f"图片预处理失败: {e}")
    
    def predict(self, image: Image.Image, return_probabilities=True):
        """
        对单张图片进行伪造检测
        
        Args:
            image: Image.Image
            return_probabilities: 是否返回概率值，否则返回类别
            
        Returns:
            如果return_probabilities=True: 返回 (real_prob, fake_prob)
            如果return_probabilities=False: 返回 "real" 或 "fake"
        """
        # 预处理图片
        image_tensor = self.preprocess_image(image)
        
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


model = ForgeryDetector()

# ------- 1) （示例）假模型 -------
def predict(img: Image.Image):
    result = model.predict(img)
    # 示例：固定返回 99% / 1%
    probs = np.array(result)
    return {"Real Image": float(probs[0]), "Fake Image": float(probs[1])}

# ------- 2) UI -------
with gr.Blocks() as demo:
    gr.Markdown("## Real-vs-Fake Image Detector")

    with gr.Row():
        # —— 左列：图片 + 按钮 ——
        with gr.Column(scale=1):
            img_input = gr.Image(label="img", type="pil")

            # 把按钮放到 **同一列** 下方
            with gr.Row():
                submit_btn = gr.Button("Submit", variant="primary")
                clear_btn = gr.ClearButton(value="Clear", components=[img_input])

        # —— 右列：结果条形图 ——
        with gr.Column(scale=1):
            label_output = gr.Label(label="output", num_top_classes=2)

    # 交互
    submit_btn.click(predict, inputs=img_input, outputs=label_output)
    clear_btn.add([label_output])  # 同时清空右侧结果

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=True)
