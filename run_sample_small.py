from app import get_local_models, initialize_model, generate_image
import os
models = get_local_models()
# pick smallest non-zero model
p=r'G:\My Drive\sd\stable-diffusion-webui\models\Stable-diffusion'
models_with_size = []
for m in models:
    fp=os.path.join(p,m)
    try:
        s=os.path.getsize(fp)
    except:
        s=0
    if s>0:
        models_with_size.append((s,m))
if not models_with_size:
    print('No usable models found')
    raise SystemExit(1)
models_with_size.sort()
model=models_with_size[0][1]
print('Using model:', model)
ok=initialize_model(model)
if not ok:
    print('Failed to initialize model')
    raise SystemExit(1)
prompt='A cinematic, photorealistic landscape, dramatic lighting, 8k'
images, info, meta = generate_image(prompt=prompt, style='Standard Quality', negative_prompt='', steps=20, cfg_scale=6.0, seed=-1, width=896, height=1152, model_name=model, num_images=2, progress=lambda *a, **k: None)
print(info)
print(meta)
os.makedirs('generated_images', exist_ok=True)
for i, img in enumerate(images):
    path=os.path.join('generated_images', f'small_sample_{i}.png')
    img.save(path)
    print('Saved', path)
