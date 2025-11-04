import os
import re
import json
import time
import hashlib
import argparse
import requests
import pytesseract



from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from PIL import Image

BASE_URL = 'http://127.0.0.1:1234/v1'
CHAT_MODEL = 'openai/gpt-oss-20b'
EMB_MODEL = 'text-embedding-nomic-embed-text-v1.5'
API_KEY = 'ignore'
INDEX_PATH = './faiss_index'
IMAGES_DIR = './images'


class LMStudioEmbeddings(Embeddings):
    '''
    Обёртка над /v1/embeddings LM Studio под интерфейс LangChain.
    Делает батчевую отправку текстов/
    '''

    def embed_documents(self, texts):
        '''
        Возвращает эмбеддинги для списка текстов.
        '''
        out = []
        for i in range(0, len(texts), 64):
            batch = texts[i:i + 64]
            payload = {'input': batch, 'model': EMB_MODEL}
            r = requests.post(
                f'{BASE_URL}/embeddings',
                headers={'Authorization': f'Bearer {API_KEY}', 'Content-Type': 'application/json'},
                data=json.dumps(payload),
                timeout=60)
            
            r.raise_for_status()
            out.extend([item['embedding'] for item in r.json()['data']])
        return out

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def absolute_src(page_url, src):
    if not src:
        return None
    src = src.strip()
    if src.startswith('//'):
        return 'https:' + src
    if src.startswith('http'):
        return src
    return urljoin(page_url, src)


def hashed_name(url):
    h = hashlib.sha256(url.encode()).hexdigest()[:20]
    ext = os.path.splitext(urlparse(url).path)[1] or '.png'
    return f'{h}{ext}'


def download_image(img_url):
    '''
    Скачивает изображение по URL в каталог IMAGES_DIR.
    '''
    os.makedirs(IMAGES_DIR, exist_ok=True)
    name = hashed_name(img_url)
    path = os.path.join(IMAGES_DIR, name)
    # если файл уже есть - повторно не качаем
    if os.path.exists(path):
        return path
    try:
        r = requests.get(img_url, timeout=20)
        if not r.ok:
            return None
        with open(path, 'wb') as f:
            f.write(r.content)
        return path
    except Exception:
        return None


def ocr_extract_text(image_path):
    '''
    Делает OCR изображения через Tesseract.
    Возвращает распознанный текст или пустую строку.
    '''
    try:
        img = Image.open(image_path)
        if img.mode not in ('RGB', 'L'):
            img = img.convert('RGB')
        text = pytesseract.image_to_string(img, lang='rus+eng', config='--psm 6')
        return text.strip()
    except Exception:
        return ''


def fetch_with_ocr(url):
    '''
    Загружает HTML-страницу, вычищает "шум"
    находит <img>, скачивает и прогоняет их через OCR,
    В конце возвращает весь текст страницы
    '''
    
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, 'html.parser')

    # удаляем шумовые теги, которые могут вносить лишний текст
    for bad in soup(['script', 'style', 'header', 'footer', 'noscript', 'nav']):
        bad.decompose()

    # собираем все картинки
    imgs = soup.find_all('img')
    ocr_results = {}

    def process_image(img):
        '''
        Функция для пула потоков: скачивает картинку и делает OCR
        '''
        src = absolute_src(url, img.get('src') or img.get('data-src'))
        if not src:
            return (None, '')
        local = download_image(src)
        if not local:
            return (src, '')
        return (src, ocr_extract_text(local))

    # параллелим скачивание и OCR картинок
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(process_image, img) for img in imgs]
        for fut in as_completed(futures):
            src, txt = fut.result()
            if src:
                ocr_results[src] = txt

    for img in imgs:
        src = absolute_src(url, img.get('src') or img.get('data-src'))
        alt = (img.get('alt') or '').strip()
        caption = ''
        fig = img.find_parent('figure')
        if fig:
            cap = fig.find('figcaption')
            if cap:
                caption = cap.get_text(' ', strip=True)
        ocr_text = ocr_results.get(src, "")
        label = (
            f"\n[IMAGE: {src or ''} | alt='{alt}" | caption="{caption}']\n"
            f"[TEXT_FROM_IMAGE: {ocr_text}]\n"
        )
        img.replace_with(label)

    text = soup.get_text('\n', strip=True)
    text = re.sub(r'\n{2,}', '\n', text)
    return text


def chunk_text(text, max_chars=1200, overlap=150):
    '''
    Нарезка текста на куски (чанки)
    '''
    paras = [p.strip() for p in text.split('\n') if p.strip()]
    chunks = []
    cur = ''
    for p in paras:
        if len(cur) + len(p) + 1 <= max_chars:
            # +1 за пробел между абзацами
            cur += ' ' + p
        else:
            chunks.append(cur.strip())
            # переносим хвост предыдущего чанка 
            cur = p[-overlap:] if overlap < len(p) else p
    if cur:
        chunks.append(cur.strip())
    return chunks


def build_faiss_index(urls):
    '''
    Для каждого URL:
      - вытягиваем текст с OCR-вставками,
      - режем на чанки,
      - упаковываем в LangChain Document с metadata={'source': url}.
    Затем строим FAISS-индекс по эмбеддингам и сохраняем локально
    '''
    corpus = []
    for url in urls:
        print(f'Обработка {url}')
        text = fetch_with_ocr(url)
        chunks = chunk_text(text)
        for chunk in chunks:
            corpus.append(Document(page_content=chunk, metadata={'source': url}))
    print(f'Создано {len(corpus)} чанков')

    embeddings = LMStudioEmbeddings()
    index = FAISS.from_documents(corpus, embeddings)
    index.save_local(INDEX_PATH)
    print(f'Индекс сохранён в {INDEX_PATH}')


def load_faiss_index():
    embeddings = LMStudioEmbeddings()
    return FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)


def chat_with_context(question, k=5):
    '''
    Ищет k наиболее похожих чанков в FAISS по эмбеддингу вопроса,
    собирает их в единый «контекст» и задаёт LM Studio вопрос с этим контекстом.
    В промпте просим модель отвечать ТОЛЬКО по контексту и использовать OCR-блоки.
    '''
    index = load_faiss_index()
    docs = index.similarity_search(question, k=k)
    context = '\n---\n'.join(d.page_content for d in docs)

    prompt = f'''
    You are a factual and concise assistant that answers questions strictly based on the provided context.
    If the answer cannot be found in the context, explicitly say you don't know.
    When relevant, use information from blocks labeled [TEXT_FROM_IMAGE:] - these contain OCR text extracted from images.
    Do not make up or assume facts not present in the context. Answer in clear, natural Russian.

    Question:
    {question}

    Context:
    {context}
    '''

    payload = {'model': CHAT_MODEL,
               'messages': [{'role': 'user', 'content': prompt}],
               'temperature': 0.2}

    r = requests.post(
        f'{BASE_URL}/chat/completions',
        headers={'Authorization': f'Bearer {API_KEY}', 'Content-Type': 'application/json'},
        data=json.dumps(payload),
        timeout=60)
    
    r.raise_for_status()
    return r.json()['choices'][0]['message']['content']



def main():
    parser = argparse.ArgumentParser(description='RAG + OCR (LM Studio + FAISS)')
    sub = parser.add_subparsers(dest='cmd')

    ing = sub.add_parser('ingest', help='Построить индекс')
    ing.add_argument('--urls', nargs='+', required=True)

    ask = sub.add_parser('ask', help='Задать вопрос')
    ask.add_argument('question', type=str)
    ask.add_argument('--k', type=int, default=5)

    args = parser.parse_args()

    if args.cmd == 'ingest':
        build_faiss_index(args.urls)
    elif args.cmd == 'ask':
        print(chat_with_context(args.question, k=args.k))
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

