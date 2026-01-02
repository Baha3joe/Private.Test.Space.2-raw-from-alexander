# ===== 必须首先导入spaces =====
try:
    import spaces
    SPACES_AVAILABLE = True
    print("✅ Spaces available - ZeroGPU mode")
except ImportError:
    SPACES_AVAILABLE = False
    print("⚠️ Spaces not available - running in regular mode")

# ===== 其他导入 =====
import os
import uuid
from datetime import datetime
import random
import torch
import gradio as gr
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
from PIL import Image
import traceback
import numpy as np

# ===== 长提示词处理 =====
try:
    from compel import Compel, ReturnedEmbeddingsType
    COMPEL_AVAILABLE = True
    print("✅ Compel available for long prompt processing")
except ImportError:
    COMPEL_AVAILABLE = False
    print("⚠️ Compel not available - using standard prompt processing")

# ===== 优化后的配置 =====
STYLE_KEYWORDS = {
    "None": {"prefix": "", "suffix": ""},
    "Realistic": {
        "prefix": "(RAW photo:1.3), (photorealistic:1.4), (hyperrealistic:1.3), 8k uhd, (ultra realistic skin texture:1.2), cinematic lighting, vibrant colors, masterpiece, realistic skin texture, detailed anatomy, professional photography",
        "suffix": "sharp focus, (everything in focus:1.3), (no bokeh:1.2), realistic skin texture, subsurface scattering, detailed anatomy, (perfect anatomy:1.2), detailed face, detailed background, lifelike, professional photography, realistic proportions, (detailed face:1.1), natural pose, expressive eyes, 8k resolution"
    },
    "Anime": {
        "prefix": "(anime style:1.3), (anime artwork:1.2), vibrant, key visual, studio anime, highly detailed anime",
        "suffix": "cel shading, clean linework, vibrant anime colors, detailed anime eyes, smooth anime skin, perfect anime proportions, manga illustration"
    },
    "Comic": {
        "prefix": "(comic book art:1.3), (graphic novel:1.2), bold inking, comic art style",
        "suffix": "bold outlines, halftone dots, pop art colors, dynamic panel, graphic illustration, cel shading, comic book style"
    },
    "Watercolor": {
        "prefix": "(watercolor painting:1.3), (watercolor art:1.2), soft edges, delicate washes, artistic",
        "suffix": "soft gradients, pastel colors, paper texture, artistic brush strokes, traditional watercolor, hand-painted"
    }
}

QUALITY_TAGS = "masterpiece, best quality, high resolution, detailed"

# 🔧 修正：使用正确的模型名称
FIXED_MODEL = "votepurchase/pornmasterPro_noobV3VAE"

# 🔧 修正：使用正确的 LoRA 配置
LORA_CONFIGS = [
    {
        "repo_id": "OedoSoldier/detail-tweaker-lora",
        "weight_name": "add_detail.safetensors",
        "adapter_name": "detail_tweaker",
        "scale": 0.8
    }
]

SAVE_DIR = "generated_images"
os.makedirs(SAVE_DIR, exist_ok=True)

# ===== 模型相关变量 =====
pipeline = None
compel_processor = None
device = None
model_loaded = False

def initialize_model():
    """优化的模型初始化 - 修复设备不一致问题"""
    global pipeline, compel_processor, device, model_loaded
    
    if model_loaded and pipeline is not None:
        print("✅ Model already loaded, skipping initialization")
        return True
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️ Using device: {device}")
        
        print(f"📦 Loading model: {FIXED_MODEL}")
        
        # 基础模型加载
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            FIXED_MODEL,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            variant="fp16" if torch.cuda.is_available() else None,
            use_safetensors=True,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        # 优化调度器
        pipeline.scheduler = EulerDiscreteScheduler.from_config(
            pipeline.scheduler.config,
            timestep_spacing="trailing"
        )
        
        # 先移到设备
        pipeline = pipeline.to(device)
        
        # 加载 LoRA
        print("🎨 Loading LoRA models...")
        adapter_names = []
        adapter_scales = []
        
        for lora_config in LORA_CONFIGS:
            try:
                print(f"   Loading: {lora_config['repo_id']}/{lora_config['weight_name']}")
                pipeline.load_lora_weights(
                    lora_config["repo_id"],
                    weight_name=lora_config["weight_name"],
                    adapter_name=lora_config["adapter_name"]
                )
                
                # 🔧 关键修复: 确保 LoRA 权重在正确的设备上
                if hasattr(pipeline, 'unet') and hasattr(pipeline.unet, 'to'):
                    pipeline.unet.to(device)
                
                adapter_names.append(lora_config["adapter_name"])
                adapter_scales.append(lora_config.get("scale", 0.8))
                print(f"   ✅ LoRA loaded: {lora_config['adapter_name']} (scale: {lora_config.get('scale', 0.8)})")
            except Exception as lora_error:
                print(f"   ⚠️ Failed to load LoRA {lora_config['adapter_name']}: {lora_error}")
                print(traceback.format_exc())
        
        # 设置 LoRA 强度
        if adapter_names:
            try:
                pipeline.set_adapters(adapter_names, adapter_weights=adapter_scales)
                print(f"✅ LoRA adapters activated with scales: {adapter_scales}")
                
                # 🔧 再次确保所有组件在同一设备
                pipeline.to(device)
            except Exception as e:
                print(f"⚠️ Failed to set adapter scales: {e}")
        
        # GPU优化
        if torch.cuda.is_available():
            try:
                pipeline.enable_vae_slicing()
                pipeline.enable_vae_tiling()
                
                try:
                    pipeline.enable_xformers_memory_efficient_attention()
                    print("✅ xFormers enabled")
                except:
                    print("⚠️ xFormers not available, using default attention")
                
                print("ℹ️ Skipping torch.compile for ZeroGPU compatibility")
                
            except Exception as opt_error:
                print(f"⚠️ Optimization warning: {opt_error}")
        
        # 初始化Compel
        if COMPEL_AVAILABLE:
            try:
                compel_processor = Compel(
                    tokenizer=[pipeline.tokenizer, pipeline.tokenizer_2],
                    text_encoder=[pipeline.text_encoder, pipeline.text_encoder_2],
                    returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                    requires_pooled=[False, True],
                    truncate_long_prompts=False
                )
                print("✅ Compel processor initialized")
            except Exception as compel_error:
                print(f"⚠️ Compel initialization failed: {compel_error}")
                compel_processor = None
        
        # 🔧 最终设备检查
        print(f"🔍 Final device check:")
        print(f"   - UNet device: {next(pipeline.unet.parameters()).device}")
        print(f"   - VAE device: {next(pipeline.vae.parameters()).device}")
        print(f"   - Text Encoder device: {next(pipeline.text_encoder.parameters()).device}")
        
        model_loaded = True
        print("✅ Model initialization complete")
        return True
        
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        print(traceback.format_exc())
        model_loaded = False
        return False

def enhance_prompt(prompt: str, style: str) -> str:
    """优化的提示词增强"""
    if not prompt or prompt.strip() == "":
        return ""
    
    style_config = STYLE_KEYWORDS.get(style, STYLE_KEYWORDS["None"])
    parts = []
    
    if style_config["prefix"]:
        parts.append(style_config["prefix"])
    
    parts.append(prompt.strip())
    
    if style_config["suffix"]:
        parts.append(style_config["suffix"])
    
    parts.append(QUALITY_TAGS)
    
    enhanced = ", ".join(parts)
    
    print(f"\n🎨 Style: {style}")
    print(f"📝 User prompt: {prompt[:100]}...")
    print(f"✨ Enhanced: {enhanced[:200]}...\n")
    
    return enhanced

def build_negative_prompt(style: str, custom_negative: str = "") -> str:
    """根据风格构建负面提示词"""
    base_negative = "(low quality:1.4), (worst quality:1.4), (bad anatomy:1.3), (bad hands:1.2), blurry, watermark, text, error, cropped, jpeg artifacts, ugly, duplicate, deformed"
    
    style_negatives = {
        "Realistic": ", (cartoon:1.3), (anime:1.3), (3d render:1.2), (illustration:1.2), (painting:1.2), (drawing:1.2), (art:1.2), (sketch:1.2), artificial, unrealistic",
        "Anime": ", (realistic:1.3), (photorealistic:1.3), (photo:1.2), (3d:1.2), (hyperrealistic:1.2)",
        "Comic": ", (realistic:1.2), (photorealistic:1.2), (blurry lines:1.2), (soft edges:1.2)",
        "Watercolor": ", (digital art:1.2), (sharp edges:1.2), (vector art:1.2), (3d:1.2)"
    }
    
    negative = base_negative
    if style in style_negatives:
        negative += style_negatives[style]
    
    if custom_negative.strip():
        negative += f", {custom_negative.strip()}"
    
    return negative

def process_with_compel(prompt, negative_prompt):
    """使用Compel处理长提示词"""
    if not compel_processor:
        return None, None
    
    try:
        conditioning, pooled = compel_processor([prompt, negative_prompt])
        print("✅ Long prompt processed with Compel")
        return conditioning, pooled
    except Exception as e:
        print(f"⚠️ Compel processing failed: {e}")
        return None, None

def apply_spaces_decorator(func):
    """应用spaces装饰器"""
    if SPACES_AVAILABLE:
        return spaces.GPU(duration=45)(func)
    return func

def create_metadata_content(prompt, enhanced_prompt, seed, steps, cfg_scale, width, height, style):
    """创建元数据"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lora_info = ", ".join([f"{lora['adapter_name']}({lora.get('scale', 1.0)})" for lora in LORA_CONFIGS])
    
    return f"""Generated Image Metadata
======================
Timestamp: {timestamp}
Original Prompt: {prompt}
Enhanced Prompt: {enhanced_prompt}
Seed: {seed}
Steps: {steps}
CFG Scale: {cfg_scale}
Dimensions: {width}x{height}
Style: {style}
LoRA: {lora_info}
"""

def cleanup_pipeline():
    """清理 pipeline 状态"""
    global pipeline
    
    if pipeline is None:
        return
    
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        
        if hasattr(pipeline, 'unet'):
            if hasattr(pipeline.unet, 'set_attn_processor'):
                try:
                    from diffusers.models.attention_processor import AttnProcessor
                    pipeline.unet.set_attn_processor(AttnProcessor())
                except:
                    pass
        
        if hasattr(pipeline, 'vae'):
            pipeline.vae.to('cpu')
            pipeline.vae.to(device)
        
        print("🧹 Pipeline cleaned")
        
    except Exception as e:
        print(f"⚠️ Cleanup warning: {e}")

@apply_spaces_decorator
def generate_image(prompt: str, style: str, negative_prompt: str = "",
                   steps: int = 25, cfg_scale: float = 7.0,
                   seed: int = -1, width: int = 1024, height: int = 1024,
                   progress=gr.Progress()):
    """图像生成主函数"""
    
    if not prompt or prompt.strip() == "":
        return None, "", "❌ Please enter a prompt"
    
    progress(0.05, desc="Initializing...")
    
    if not initialize_model():
        return None, "", "❌ Failed to load model"
    
    cleanup_pipeline()
    
    progress(0.1, desc="Processing prompt...")
    
    try:
        if seed == -1:
            seed = random.randint(0, np.iinfo(np.int32).max)
        
        generator = torch.Generator(device).manual_seed(seed)
        
        enhanced_prompt = enhance_prompt(prompt, style)
        final_negative = build_negative_prompt(style, negative_prompt)
        
        print(f"🔧 Generation params: seed={seed}, steps={steps}, cfg={cfg_scale}, size={width}x{height}")
        print(f"📝 Prompt preview: {enhanced_prompt[:100]}...")
        
        progress(0.2, desc="Generating image...")
        
        prompt_length = len(enhanced_prompt.split())
        use_compel = prompt_length > 50 and compel_processor is not None
        
        if use_compel:
            print(f"📏 Long prompt detected ({prompt_length} words), using Compel")
            conditioning, pooled = process_with_compel(enhanced_prompt, final_negative)
            
            if conditioning is not None:
                result = pipeline(
                    prompt_embeds=conditioning[0:1],
                    pooled_prompt_embeds=pooled[0:1],
                    negative_prompt_embeds=conditioning[1:2],
                    negative_pooled_prompt_embeds=pooled[1:2],
                    num_inference_steps=steps,
                    guidance_scale=cfg_scale,
                    width=width,
                    height=height,
                    generator=generator,
                    output_type="pil"
                ).images[0]
            else:
                print("⚠️ Falling back to standard generation")
                result = pipeline(
                    prompt=enhanced_prompt,
                    negative_prompt=final_negative,
                    num_inference_steps=steps,
                    guidance_scale=cfg_scale,
                    width=width,
                    height=height,
                    generator=generator,
                    output_type="pil"
                ).images[0]
        else:
            print(f"📝 Standard generation ({prompt_length} words)")
            result = pipeline(
                prompt=enhanced_prompt,
                negative_prompt=final_negative,
                num_inference_steps=steps,
                guidance_scale=cfg_scale,
                width=width,
                height=height,
                generator=generator,
                output_type="pil"
            ).images[0]
        
        progress(0.95, desc="Finalizing...")
        
        if not isinstance(result, Image.Image):
            if isinstance(result, np.ndarray):
                if result.dtype != np.uint8:
                    result = (result * 255).astype(np.uint8)
                result = Image.fromarray(result)
        
        metadata = create_metadata_content(
            prompt, enhanced_prompt, seed, steps, cfg_scale,
            width, height, style
        )
        
        generation_info = f"Style: {style} | Seed: {seed} | Size: {width}×{height} | Steps: {steps} | CFG: {cfg_scale}"
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        progress(1.0, desc="Complete!")
        print("✅ Generation successful\n")
        
        return result, generation_info, metadata
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Generation error: {error_msg}")
        print(traceback.format_exc())
        
        try:
            cleanup_pipeline()
        except:
            pass
        
        return None, "", f"❌ Generation failed: {error_msg}"

# ===== CSS样式 =====
css = """
.gradio-container {
    max-width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
    background: linear-gradient(135deg, #e6a4f2 0%, #1197e4 100%) !important;
    min-height: 100vh !important;
    font-family: 'Segoe UI', Arial, sans-serif !important;
}

.main-content {
    background: rgba(255, 255, 255, 0.9) !important;
    border-radius: 20px !important;
    padding: 20px !important;
    margin: 15px !important;
    box-shadow: 0 10px 25px rgba(255, 255, 255, 0.2) !important;
    min-height: calc(100vh - 30px) !important;
    color: #3e3e3e !important;
    backdrop-filter: blur(10px) !important;
}

.title {
    text-align: center !important;
    background: linear-gradient(45deg, #bb6ded, #08676b) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    font-size: 2rem !important;
    margin-bottom: 15px !important;
    font-weight: bold !important;
}

.warning-box {
    background: linear-gradient(45deg, #bb6ded, #08676b) !important;
    color: white !important;
    padding: 8px !important;
    border-radius: 8px !important;
    margin-bottom: 15px !important;
    text-align: center !important;
    font-weight: bold !important;
    font-size: 14px !important;
}

.prompt-box textarea, .prompt-box input {
    border-radius: 10px !important;
    border: 2px solid #bb6ded !important;
    padding: 15px !important;
    font-size: 18px !important;
    background: linear-gradient(135deg, rgba(245, 243, 255, 0.9), rgba(237, 233, 254, 0.9)) !important;
    color: #2d2d2d !important;
}

.prompt-box textarea:focus, .prompt-box input:focus {
    border-color: #08676b !important;
    box-shadow: 0 0 15px rgba(77, 8, 161, 0.3) !important;
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.95), rgba(248, 249, 250, 0.95)) !important;
}

.controls-section {
    background: linear-gradient(135deg, rgba(224, 218, 255, 0.8), rgba(196, 181, 253, 0.8)) !important;
    border-radius: 12px !important;
    padding: 15px !important;
    margin-bottom: 8px !important;
    border: 2px solid rgba(187, 109, 237, 0.3) !important;
    backdrop-filter: blur(5px) !important;
}

.controls-section label {
    font-weight: 600 !important;
    color: #2d2d2d !important;
    margin-bottom: 8px !important;
}

.controls-section input[type="radio"] {
    accent-color: #bb6ded !important;
}

.controls-section input[type="number"],
.controls-section input[type="range"] {
    background: rgba(255, 255, 255, 0.9) !important;
    border: 1px solid #bb6ded !important;
    border-radius: 6px !important;
    padding: 8px !important;
    color: #2d2d2d !important;
}

.generate-btn {
    background: linear-gradient(45deg, #bb6ded, #08676b) !important;
    color: white !important;
    border: none !important;
    padding: 15px 25px !important;
    border-radius: 25px !important;
    font-size: 16px !important;
    font-weight: bold !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}

.generate-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(187, 109, 237, 0.5) !important;
}

.image-output {
    border-radius: 15px !important;
    overflow: hidden !important;
    max-width: 100% !important;
    max-height: 70vh !important;
    border: 3px solid #08676b !important;
    box-shadow: 0 8px 20px rgba(0,0,0,0.15) !important;
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.9), rgba(248, 249, 250, 0.9)) !important;
}

.image-info {
    background: linear-gradient(135deg, rgba(248, 249, 250, 0.2), rgba(233, 236, 239, 0.9)) !important;
    border-radius: 8px !important;
    padding: 12px !important;
    margin-top: 10px !important;
    font-size: 12px !important;
    color: #495057 !important;
    border: 2px solid rgba(187, 109, 237, 0.2) !important;
    backdrop-filter: blur(5px) !important;
}

.metadata-box {
    background: linear-gradient(135deg, rgba(248, 249, 250, 0.2), rgba(233, 236, 239, 0.9)) !important;
    border-radius: 8px !important;
    padding: 15px !important;
    margin-top: 15px !important;
    font-family: 'Courier New', monospace !important;
    font-size: 12px !important;
    color: #495057 !important;
    border: 2px solid rgba(187, 109, 237, 0.2) !important;
    backdrop-filter: blur(5px) !important;
    white-space: pre-wrap !important;
    overflow-y: auto !important;
    max-height: 300px !important;
}

@media (max-width: 768px) {
    .main-content {
        margin: 10px !important;
        padding: 15px !important;
    }
    .title {
        font-size: 1.5rem !important;
    }
}
"""

# ===== 创建UI =====
def create_interface():
    with gr.Blocks(css=css, title="Adult NSFW AI Image Generator") as interface:
        with gr.Column(elem_classes=["main-content"]):
            gr.HTML('<div class="title">Adult NSFW AI Image Generator</div>')
            gr.HTML('<div class="warning-box">⚠️ 18+ CONTENT WARNING ⚠️</div>')
            
            with gr.Row():
                with gr.Column(scale=2):
                    prompt_input = gr.Textbox(
                        label="Detailed Prompt",
                        placeholder="Enter your detailed prompt here...",
                        lines=15,
                        elem_classes=["prompt-box"]
                    )
                    
                    negative_prompt_input = gr.Textbox(
                        label="Negative Prompt (Optional)",
                        placeholder="Additional things you don't want...",
                        lines=4,
                        elem_classes=["prompt-box"]
                    )
                
                with gr.Column(scale=1):
                    with gr.Group(elem_classes=["controls-section"]):
                        style_input = gr.Radio(
                            label="Style Preset",
                            choices=list(STYLE_KEYWORDS.keys()),
                            value="Realistic"
                        )
                    
                    with gr.Group(elem_classes=["controls-section"]):
                        seed_input = gr.Number(
                            label="Seed (-1 for random)",
                            value=-1,
                            precision=0
                        )
                    
                    with gr.Group(elem_classes=["controls-section"]):
                        width_input = gr.Slider(
                            label="Width",
                            minimum=512,
                            maximum=2048,
                            value=1024,
                            step=64
                        )
                    
                    with gr.Group(elem_classes=["controls-section"]):
                        height_input = gr.Slider(
                            label="Height",
                            minimum=512,
                            maximum=2048,
                            value=1024,
                            step=64
                        )
                    
                    with gr.Group(elem_classes=["controls-section"]):
                        steps_input = gr.Slider(
                            label="Steps",
                            minimum=10,
                            maximum=50,
                            value=25,
                            step=1
                        )
                        
                        cfg_input = gr.Slider(
                            label="CFG Scale",
                            minimum=1.0,
                            maximum=15.0,
                            value=7.0,
                            step=0.1
                        )
                    
                    generate_button = gr.Button(
                        "GENERATE",
                        elem_classes=["generate-btn"],
                        variant="primary"
                    )
            
            image_output = gr.Image(
                label="Generated Image",
                elem_classes=["image-output"],
                show_label=False,
                container=True
            )
            
            with gr.Row():
                generation_info = gr.Textbox(
                    label="Generation Info",
                    interactive=False,
                    elem_classes=["image-info"],
                    show_label=True,
                    visible=False
                )
            
            with gr.Row():
                metadata_display = gr.Textbox(
                    label="Image Metadata",
                    interactive=True,
                    elem_classes=["metadata-box"],
                    show_label=True,
                    lines=15,
                    visible=False
                )
            
            def on_generate(prompt, style, neg_prompt, steps, cfg, seed, width, height):
                image, info, metadata = generate_image(
                    prompt, style, neg_prompt, steps, cfg, seed, width, height
                )
                
                if image is not None:
                    return (
                        image,
                        info,
                        metadata,
                        gr.update(visible=True, value=info),
                        gr.update(visible=True, value=metadata)
                    )
                else:
                    return (
                        None,
                        info,
                        "",
                        gr.update(visible=False),
                        gr.update(visible=False)
                    )
            
            generate_button.click(
                fn=on_generate,
                inputs=[
                    prompt_input, style_input, negative_prompt_input,
                    steps_input, cfg_input, seed_input, width_input, height_input
                ],
                outputs=[
                    image_output, generation_info, metadata_display,
                    generation_info, metadata_display
                ],
                show_progress=True
            )
            
            prompt_input.submit(
                fn=on_generate,
                inputs=[
                    prompt_input, style_input, negative_prompt_input,
                    steps_input, cfg_input, seed_input, width_input, height_input
                ],
                outputs=[
                    image_output, generation_info, metadata_display,
                    generation_info, metadata_display
                ],
                show_progress=True
            )
    
    return interface

# ===== 启动应用 =====
if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 Starting NSFW Image Generator")
    print("="*50)
    print(f"📦 Model: {FIXED_MODEL}")
    print(f"🖥️ Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"⚡ ZeroGPU: {'Enabled' if SPACES_AVAILABLE else 'Disabled'}")
    print(f"📝 Compel: {'Available' if COMPEL_AVAILABLE else 'Not Available'}")
    print(f"🎨 LoRA: detail-tweaker-lora (scale: {LORA_CONFIGS[0].get('scale', 0.8)})")
    print("="*50 + "\n")
    
    app = create_interface()
    app.queue(max_size=10, default_concurrency_limit=2)
    
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )