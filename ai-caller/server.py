"""ИИ-менеджер TenderMax — стенд для обкатки.

Речь клиента -> K5 (распознавание казахского/русского) -> Claude (ответ) -> MMS (озвучка kk/ru).
Запуск: python server.py   затем открыть http://localhost:8800
Ключ Claude: переменная окружения ANTHROPIC_API_KEY или файл .anthropic_key рядом с этим файлом.
"""
import os, io, re, json, time, uuid, base64, subprocess, threading
import numpy as np, soundfile as sf, torch, requests
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from transformers import AutoModel, VitsModel, AutoTokenizer

B = os.path.dirname(os.path.abspath(__file__))
MODELS = os.environ.get('MODELS_DIR', os.path.join(B, 'models'))
SR = 16000
LOG = os.path.join(B, 'sessions')
os.makedirs(LOG, exist_ok=True)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', DEV, flush=True)

k5 = AutoModel.from_pretrained(os.path.join(MODELS, 'k5'), trust_remote_code=True).eval()
try:
    k5 = k5.to(DEV)
except Exception as e:  # K5 falls back to CPU if its code does not support the device
    print('K5 on cpu:', e, flush=True)
tts = {}
for lang in ('kaz', 'rus'):
    tts[lang] = (VitsModel.from_pretrained(os.path.join(MODELS, f'mms-{lang}')).eval().to(DEV),
                 AutoTokenizer.from_pretrained(os.path.join(MODELS, f'mms-{lang}')))
lock = threading.Lock()

KEY = os.environ.get('ANTHROPIC_API_KEY') or open(os.path.join(B, '.anthropic_key')).read().strip()
MODEL = os.environ.get('BOT_MODEL', 'claude-sonnet-5-5')
THINK = {'type': 'disabled'} if 'haiku' in MODEL else {'type': 'between_tools'}

KAZ_CHARS = set('әғқңөұүһіӘҒҚҢӨҰҮҺІ')
FIX = {'TenderMax': 'Тендермакс', 'Tendermax': 'Тендермакс', 'WhatsApp': 'Ватсап', 'Kaspi': 'Каспи',
       'Halyk': 'Халық', 'VIP': 'вип', 'Zoom': 'Зум', 'MPKZ': 'эм пэ кэ зэ', 'ETS Tender': 'ЕТС тендер',
       'ETS': 'ЕТС', 'Air Astana': 'Эйр Астана', 'Astana Motors': 'Астана Моторс', 'Pro': 'Про', 'Start': 'Старт'}


def system_prompt():
    return open(os.path.join(B, 'prompt.md'), encoding='utf-8').read().replace(
        '{offer}', open(os.path.join(B, 'offer.md'), encoding='utf-8').read())


def stt(path):
    pcm = subprocess.run(['ffmpeg', '-loglevel', 'error', '-i', path, '-ac', '1', '-ar', str(SR), '-f', 's16le', '-'],
                         capture_output=True, check=True).stdout
    a = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768
    out = []
    for i in range(0, len(a), 25 * SR):
        piece = a[i:i + 25 * SR]
        if len(piece) < 0.4 * SR:
            continue
        p = os.path.join(LOG, f'tmp_{uuid.uuid4().hex}.wav')
        sf.write(p, piece, SR)
        with lock:
            out.append(k5.transcribe(p).text.strip())
        os.remove(p)
    return ' '.join(x for x in out if x)


def speak(text):
    for a, b in FIX.items():
        text = text.replace(a, b)
    lang = 'kaz' if any(c in KAZ_CHARS for c in text) else 'rus'
    model, tok = tts[lang]
    wav = []
    for p in [p for p in re.split(r'(?<=[.!?])\s+', text) if p.strip()]:
        ids = tok(p, return_tensors='pt').to(DEV)
        if ids['input_ids'].shape[1] < 2:
            continue
        with lock, torch.no_grad():
            w = model(**ids).waveform[0].cpu().numpy()
        wav += [w, np.zeros(int(0.25 * model.config.sampling_rate), np.float32)]
    if not wav:
        return None
    buf = io.BytesIO()
    sf.write(buf, np.concatenate(wav), model.config.sampling_rate, format='WAV')
    return base64.b64encode(buf.getvalue()).decode()


def llm(history):
    msgs = [{'role': 'user', 'content': '[Звонок начался, клиент взял трубку]'}]
    for r, t in history:
        msgs.append({'role': 'user' if r == 'client' else 'assistant', 'content': t})
    r = requests.post('https://api.anthropic.com/v1/messages', timeout=60, headers={
        'x-api-key': KEY, 'anthropic-version': '2023-06-01', 'content-type': 'application/json'},
        json={'model': MODEL, 'max_tokens': 400, 'thinking': THINK,
              'system': [{'type': 'text', 'text': system_prompt(), 'cache_control': {'type': 'ephemeral'}}],
              'messages': msgs})
    r.raise_for_status()
    txt = ' '.join(b['text'] for b in r.json()['content'] if b.get('type') == 'text').strip()
    return re.sub(r'^(Айгерім|Айгерим)\s*:\s*', '', txt)


SESS = {}
app = FastAPI()


def save(sid):
    json.dump(SESS[sid], open(os.path.join(LOG, f'{sid}.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


@app.get('/', response_class=HTMLResponse)
def index():
    return open(os.path.join(B, 'index.html'), encoding='utf-8').read()


@app.post('/api/start')
def start(lang: str = Form('kk'), name: str = Form('')):
    sid = uuid.uuid4().hex[:10]
    n = f', {name}' if name else ''
    if lang == 'ru':
        g = f'Здравствуйте{n}! Меня зовут Айгерим, я звоню из обучающего центра Тендермакс. Вам удобно сейчас говорить?'
    else:
        g = f'Сәлеметсіз бе{n}! Менің атым Айгерім, мен Тендермакс оқу орталығынан хабарласып тұрмын. Қазір сөйлесуге ыңғайлы ма?'
    SESS[sid] = {'lang': lang, 'name': name, 'history': [['bot', g]], 'turns': []}
    t0 = time.time()
    audio = speak(g)
    save(sid)
    return {'sid': sid, 'bot_text': g, 'audio': audio, 't_tts': round(time.time() - t0, 2)}


def respond(sid, client_text, t_stt):
    s = SESS[sid]
    s['history'].append(['client', client_text or '[тишина / не расслышано]'])
    t0 = time.time()
    reply = llm(s['history'])
    t_llm = time.time() - t0
    end = '[END]' in reply
    reply = reply.replace('[END]', '').strip()
    s['history'].append(['bot', reply])
    t0 = time.time()
    audio = speak(reply)
    turn = {'client': client_text, 'bot': reply, 't_stt': round(t_stt, 2), 't_llm': round(t_llm, 2),
            't_tts': round(time.time() - t0, 2), 'end': end}
    s['turns'].append(turn)
    save(sid)
    return {**turn, 'audio': audio}


@app.post('/api/turn')
async def turn(sid: str = Form(...), audio: UploadFile = File(...)):
    p = os.path.join(LOG, f'in_{uuid.uuid4().hex}')
    open(p, 'wb').write(await audio.read())
    t0 = time.time()
    text = stt(p)
    os.remove(p)
    return respond(sid, text, time.time() - t0)


@app.post('/api/text')
def text_turn(sid: str = Form(...), text: str = Form(...)):
    return respond(sid, text, 0.0)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8800)
