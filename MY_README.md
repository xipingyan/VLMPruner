# MY README

Refer README.md to setup <br>

Refer https://github.com/xipingyan/VLMPruner?tab=readme-ov-file#llava-15-7b13b  <br>

Note: download eval.zip, OCRBench Images and OCRBench json, download model: clip-vit-large-patch14-336, llava-v1.5-7b. <br>

move model to:
```
models/
├── clip-vit-large-patch14-336
├── llava-v1.5-7b
```

```
source python-env/bin/activate
bash ./scripts/v1_5/eval/ocrbench.sh 0.889 0.8 16
```