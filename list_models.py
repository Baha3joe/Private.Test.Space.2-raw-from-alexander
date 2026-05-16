import os
p=r'G:\My Drive\sd\stable-diffusion-webui\models\Stable-diffusion'
files=[f for f in os.listdir(p) if os.path.splitext(f)[1].lower() in ['.safetensors','.ckpt']]
for f in sorted(files):
    fp=os.path.join(p,f)
    try:
        s=os.path.getsize(fp)
    except Exception as e:
        s=0
    print(f, s//(1024*1024), 'MB')
