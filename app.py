# https://github.com/gpt4free/gpt4free.github.io/blob/main/docs%2Fproviders-and-models.md
# https://github.com/gpt4free/gpt4free.github.io/blob/main/docs/authentication.md
# https://openai-docs.ru/docs/guides/text-generation#completions-api

import base64
import os
import json
import time
import traceback
import logging
from logging.config import dictConfig

# Existing Flask imports, now including render_template for the image generation part
from flask import Flask, Response, request, jsonify, session, render_template, url_for

from werkzeug.middleware.proxy_fix import ProxyFix  # Импортируем ProxyFix

# G4F client imports
from g4f import Provider as Providers
from g4f.client import Client

from bingart import BingArt

import tiktoken  # pip install tiktoken
import time

import json

import traceback

app = Flask(__name__)

# Применяем ProxyFix к WSGI-приложению Flask.
# x_for=1: Обрабатывает X-Forwarded-For (IP клиента)
# x_host=1: Обрабатывает X-Forwarded-Host (оригинальный Host)
# x_proto=1: Обрабатывает X-Forwarded-Proto (HTTP/HTTPS)
# x_prefix=1: САМЫЙ ВАЖНЫЙ ПАРАМЕТР ДЛЯ ВАШЕГО СЛУЧАЯ.
#             Он заставляет ProxyFix искать заголовок X-Forwarded-Prefix
#             (или аналогичные, которые может добавить прокси) и
#             использовать его для установки request.script_root.
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_host=1, x_proto=1, x_prefix=1, x_port=1)

client = Client()

MAX_TOKENS_CONTEXT = 1000000  # пока так, по идее, нужно заменить на словарь

# --- Настройка сессий для генерации изображений (из generate-image.py) ---
# Для сохранения данных между запросами (например, prompt и URL изображений)
app.config[
    'SESSION_TYPE'] = 'filesystem'  # Тип хранилища сессий: файловая система
app.secret_key = os.urandom(
    24
)  # Секретный ключ для шифрования данных сессии. Обязательно для работы сессий!
# -------------------------------------------------------------------------
'''
Все логи с уровнем INFO и выше будут выводиться в консоль (где запущено приложение).
Все логи с уровнем ERROR и выше будут записываться в файл error.log.
'''
dictConfig({
    'version': 1,  # версия схемы конфигурации (обязательно 1)
    'formatters': {  # как форматировать сообщения в логах
        'default': {
            'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
            # Пример сообщения: [2025-06-15 01:23:45] ERROR in app: Ошибка подключения
        }
    },
    'handlers': {  # куда писать логи
        'wsgi': {  # обработчик для вывода логов в консоль (stdout/stderr)
            'class': 'logging.StreamHandler',
            'stream':
            'ext://flask.logging.wsgi_errors_stream',  # поток ошибок Flask
            'formatter': 'default'  # использовать формат из formatters.default
        },
        'file': {  # обработчик для записи логов в файл
            'class': 'logging.FileHandler',
            'filename': 'error.log',  # имя файла для логов
            'formatter': 'default',
            'level':
            'ERROR',  # записывать только ошибки и выше (ERROR, CRITICAL)
        }
    },
    'root': {  # корневой логгер (главный)
        'level': 'INFO',  # минимальный уровень логирования (INFO и выше)
        'handlers':
        ['wsgi', 'file']  # использовать оба обработчика: в консоль и в файл
    }
})

with open('error.log', 'w'):
    pass  # файл будет очищен (обрезан до 0 байт)


def count_tokens(messages, model="gpt-4o-mini"):
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    total_tokens = 0
    for msg in messages:
        total_tokens += 4 + len(enc.encode(msg.get("role", ""))) + len(
            enc.encode(msg.get("content", "")))
    total_tokens += 2
    return total_tokens


def trim_messages_to_fit(messages,
                         max_tokens=MAX_TOKENS_CONTEXT,
                         model="gpt-4o-mini"):
    while count_tokens(messages, model) > max_tokens and len(messages) > 1:
        messages.pop(0)
    return messages


def parse_model_string(model_string):

    def resolve_provider(name):
        return getattr(Providers, name, None)

    parts = model_string.split('#')

    provider_cls = None
    model = None
    settings = {}
    settings_str = None

    if len(parts) == 1:
        model = parts[0]

    elif len(parts) == 2:
        # model#settings OR provider#model
        if parts[1].startswith('{'):
            model = parts[0]
            settings_str = parts[1]
        else:
            provider_cls = resolve_provider(parts[0])
            model = parts[1]

    elif len(parts) == 3:
        provider_cls = resolve_provider(parts[0])
        model = parts[1]
        settings_str = parts[2]

    # Обработка settings, если задан
    if settings_str:
        # Попробуем декодировать base64
        if len(settings_str) % 4 == 0 and settings_str.replace(
                '-', '').replace('_', '').isalnum():
            try:
                decoded = base64.b64decode(settings_str).decode('utf-8')
                if decoded.strip().startswith('{'):
                    settings_str = decoded
            except Exception:
                pass
        # Попробуем распарсить JSON
        try:
            # Попытка двойной десериализации на случай экранированных строк
            settings = json.loads(settings_str)
            if isinstance(settings, str) and settings.strip().startswith('{'):
                settings = json.loads(settings)
        except Exception:
            settings = {}

    return {"provider": provider_cls, "model": model, "settings": settings}


# Пример curl для чата:
# curl -X POST http://localhost:3003/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"PollinationsAI:mistral\",\"messages\":[{\"role\":\"user\",\"content\":\"Кто ты?\"}]}"
# curl -X POST http://localhost:3003/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"claude-3.7-sonnet\",\"messages\":[{\"role\":\"user\",\"content\":\"Кто ты?\"}]}"
@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    data = request.get_json()
    model_string = data.get("model", "gpt-4o-mini")
    messages = data.get("messages", [])
    max_tokens = data.get("max_tokens", MAX_TOKENS_CONTEXT)
    stream = data.get("stream", False)
    debug_serialized_data = json.dumps(
        data, indent=2, ensure_ascii=False)  # for debug, don't delete it
    if not isinstance(messages, list) or len(messages) == 0:
        return jsonify({"error": "messages must be a non-empty list"}), 400

    # Parse the model string
    parsed_model = parse_model_string(model_string)
    model_name = parsed_model["model"]
    provider = parsed_model["provider"]
    settings = parsed_model["settings"]

    messages = trim_messages_to_fit(messages, MAX_TOKENS_CONTEXT, model_name)

    # часть моделей не читает дальше 1го сообщения
    # Если включён режим one_message, объединяем все сообщения в одно
    if settings.get("one_message", False):
        combined_content = ""
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            combined_content += f"{role}: {content}\n"
        messages = [{"role": "user", "content": combined_content.strip()}]

    try:
        # Prepare kwargs for create method
        kwargs = {
            "model": model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": stream
        }

        # Add provider if specified
        if provider:
            kwargs["provider"] = provider

        # Add any additional settings
        kwargs.update(settings)

        if stream:
            # Создаём генератор для стриминга
            response_generator = client.chat.completions.create(**kwargs)

            def generate():
                # full_content = [] # Это не нужно для стриминга, клиент сам собирает
                for chunk in response_generator:
                    try:
                        # Используем model_dump_json() для получения JSON-строки,
                        # соответствующей формату OpenAI ChatCompletionChunk
                        # exclude_unset=True помогает избежать отправки полей, которые не были установлены,
                        # что типично для промежуточных чанков.
                        yield f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"
                    except Exception as e:
                        # Логирование ошибок при обработке чанка, но не прерываем поток
                        print(f"Error processing chunk: {e}")
                        continue

                # Сигнал [DONE] должен быть отправлен как отдельная строка.
                # Он не должен быть внутри JSON-объекта, как это было в вашем примере.
                yield "data: [DONE]\n\n"

            return Response(generate(), mimetype="text/event-stream")
        else:
            # Обработка не-стримингового запроса
            response = client.chat.completions.create(**kwargs)

            return jsonify({
                "id":
                f"chatcmpl-{int(time.time())}",
                "object":
                "chat.completion",
                "created":
                int(time.time()),
                "model":
                model_string,
                "choices": [{
                    "index":
                    0,  # Добавлено согласно спецификации OpenAI
                    "message": {
                        "role": "assistant",
                        "content": response.choices[0].message.content
                    },
                    "finish_reason":
                    response.choices[0].finish_reason if hasattr(
                        response.choices[0], 'finish_reason') else
                    "stop"  # Учитываем finish_reason из ответа
                }],
                "usage":
                response.usage.model_dump()
                if hasattr(response.usage, "model_dump") else {
                }  # Правильно получаем usage
            })
    except Exception as e:
        # Для отладки можно выводить больше информации об ошибке
        print(f"Error in chat_completions: {e}", exc_info=True)
        app.logger.error(
            f"Error in chat_completions: {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


# Пример curl для автодополнения кода:
# curl -X POST http://localhost:3003/v1/completions -H "Content-Type: application/json" -d "{\"model\":\"gpt-4o-mini\",\"prompt\":\"def hello_world():\\n    \",\"max_tokens\":50,\"temperature\":0.5}"
@app.route("/v1/completions", methods=["POST"])
def completions():
    data = request.get_json()
    model_string = data.get("model", "gpt-4o-mini")
    prompt = data.get("prompt", "")
    max_tokens = data.get("max_tokens", 100)
    temperature = data.get("temperature", 0.7)
    stop = data.get("stop", None)

    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    parsed = parse_model_string(model_string)
    model_name = parsed["model"]
    provider = parsed["provider"]
    settings = parsed["settings"]

    # Wrap prompt in chat format
    messages = [{"role": "user", "content": prompt}]

    try:
        prompt_tokens = count_tokens(messages, model_name)
        kwargs = {
            "model": model_name,
            "messages": messages,
            "stream": False,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        if stop:
            kwargs["stop"] = stop
        if provider:
            kwargs["provider"] = provider
        kwargs.update(settings)

        response = client.chat.completions.create(**kwargs)
        completion = response.choices[0].message.content
        completion_tokens = count_tokens([{
            "role": "assistant",
            "content": completion
        }], model_name)

        # Формируем ответ в соответствии со спецификацией Completions API
        result = {
            "id":
            f"cmpl-{int(time.time())}",
            "object":
            "text_completion",
            "created":
            int(time.time()),
            "model":
            model_string,
            "choices": [{
                "index":
                0,
                "text":
                completion,
                "logprobs":
                None,
                "finish_reason":
                response.choices[0].finish_reason if hasattr(
                    response.choices[0], 'finish_reason') else "stop"
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        }

        return jsonify(result)

    except Exception as e:
        print(f"Error in completions: {e}", exc_info=True)
        app.logger.error(
            f"Error in chat_completions: {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


# --- Функции и инициализация для генерации изображений (из generate-image.py) ---
def load_cookies(file_path):
    """Загружает куки из JSON файла."""
    try:
        # Полный путь к файлу куки
        script_dir = os.path.dirname(__file__)
        full_file_path = os.path.join(script_dir, file_path)

        with open(full_file_path, 'r') as f:
            cookies = json.load(f)
            # Удаляем пустые значения куки
            return {k: v for k, v in cookies.items() if v}
    except FileNotFoundError:
        print(f"Файл не найден: {full_file_path}"
              )  # Выводим полный путь для отладки
        return {}
    except json.JSONDecodeError:
        print(f"Ошибка декодирования JSON в файле: {full_file_path}")
        return {}


# Загрузка куки и инициализация BingArt клиента при старте приложения
cookies = load_cookies("cookies.json")
bing_art = BingArt(auth_cookie_U=cookies.get('_U'),
                   auth_cookie_KievRPSSecAuth=cookies.get('KievRPSSecAuth'))
# ---------------------------------------------------------------------------------

# --- Новые эндпоинты для генерации изображений ---


@app.route('/images')
def images():
    """Отображение изображений."""
    prompt = session.get('prompt', '')
    image_urls = session.get('image_urls', [])
    error_message = session.get('error_message', None)
    session.pop('error_message', None)

    # url_for(_external=True) сгенерирует полный URL, автоматически включая
    # request.script_root, если он установлен.
    generate_images_url = url_for('generate_images', _external=True)
    return render_template(
        'index.html',
        image_urls=image_urls,
        prompt=prompt,
        error_message=error_message,
        generate_images_url=generate_images_url)  # <--- Добавлено


@app.route('/generate-images', methods=['POST'])
def generate_images():
    """Генерация изображений с использованием bingart."""
    prompt = request.form.get('prompt')
    session['prompt'] = prompt
    image_urls = []
    error_message = None

    # При каждой генерации изображений создаем новый объект BingArt
    # с помощью функции load_cookies, чтобы убедиться, что куки актуальны.
    # Это позволяет избежать проблем с устаревшими сессиями.
    # Важно: close_session() вызывается в finally, чтобы гарантировать закрытие сессии.
    local_bing_art = BingArt(
        auth_cookie_U=cookies.get('_U'),
        auth_cookie_KievRPSSecAuth=cookies.get('KievRPSSecAuth'))

    try:
        if prompt:
            results = local_bing_art.generate_images(prompt)
            if results and 'images' in results:
                image_urls = [
                    img['url'] for i, img in enumerate(results['images'])
                    if i % 2 == 0
                ][:4]
            else:
                error_message = "Не удалось получить изображения."
        else:
            error_message = "Коммандер, введи промпт!"
    except Exception as e:
        error_message = f"Произошла ошибка: {str(e)}"
        print(f"Ошибка при генерации изображения: {e}")
    finally:
        # Важно: закрываем сессию BingArt после использования
        local_bing_art.close_session()

    if error_message:
        session['error_message'] = error_message
        return jsonify({'error': error_message}), 500

    session['image_urls'] = image_urls
    return jsonify({'image_urls': image_urls})


if __name__ == "__main__":
    #app.run(port=3003, debug=True)
    port = int(os.environ.get("PORT", 3003))
    app.run(host="0.0.0.0", port=port)
