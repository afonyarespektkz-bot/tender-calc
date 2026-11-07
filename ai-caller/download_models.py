"""Скачивает модели: K5 (распознавание kk/ru, ~2.3 ГБ) и MMS (озвучка kk/ru, по ~145 МБ)."""
import os
from huggingface_hub import hf_hub_download

B = os.path.dirname(os.path.abspath(__file__))
M = os.environ.get('MODELS_DIR', os.path.join(B, 'models'))
SETS = {
    'k5': ('alibiserikbay/k5-kazakh-russian-stt', ['config.json', 'modeling_gigaam.py', 'model.safetensors']),
    'mms-kaz': ('facebook/mms-tts-kaz', ['config.json', 'model.safetensors', 'tokenizer_config.json', 'vocab.json',
                                         'special_tokens_map.json']),
    'mms-rus': ('facebook/mms-tts-rus', ['config.json', 'model.safetensors', 'tokenizer_config.json', 'vocab.json',
                                         'special_tokens_map.json']),
}
for name, (repo, files) in SETS.items():
    d = os.path.join(M, name)
    os.makedirs(d, exist_ok=True)
    for f in files:
        print('->', repo, f, flush=True)
        hf_hub_download(repo, f, local_dir=d)
print('Готово:', M)
