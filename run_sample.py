from app import LOCAL_MODEL_DIRECTORY, get_local_models, initialize_model, generate_image
import os

models = get_local_models()
if not models:
    print('No local models found')
    raise SystemExit(1)

models_with_size = []
for model_name in models:
    model_path = os.path.join(LOCAL_MODEL_DIRECTORY, model_name)
    try:
        size = os.path.getsize(model_path)
    except OSError:
        size = float('inf')
    if size >= 50 * 1024 * 1024:
        models_with_size.append((size, model_name))

if not models_with_size:
    print('No valid model files found with size >= 50MB')
    raise SystemExit(1)

model = min(models_with_size, key=lambda x: x[0])[1]
print('Using model:', model)

ok = initialize_model(model)
if not ok:
    print('Failed to initialize model')
    raise SystemExit(1)

prompt = "A cinematic, photorealistic landscape, dramatic lighting, 8k"

images, info, meta = generate_image(
    prompt=prompt,
    style='Standard Quality',
    negative_prompt='',
    steps=20,
    cfg_scale=6.0,
    seed=-1,
    width=896,
    height=1152,
    model_name=model,
    num_images=2,
    progress=lambda *a, **k: None
)

print(info)
print(meta)

os.makedirs('generated_images', exist_ok=True)
for i, img in enumerate(images):
    path = os.path.join('generated_images', f'sample_{i}.png')
    img.save(path)
    print('Saved', path)
