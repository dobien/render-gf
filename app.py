# https://github.com/gpt4free/gpt4free.github.io/blob/main/docs%2Fproviders-and-models.md
# https://github.com/gpt4free/gpt4free.github.io/blob/main/docs/authentication.md
# https://openai-docs.ru/docs/guides/text-generation#completions-api

import base64
import os
# Отключить проверку SSL для всего requests/HF-клиента
#os.environ["CURL_CA_BUNDLE"] = ""
#os.environ["REQUESTS_CA_BUNDLE"] = ""
from g4f import Provider as Providers  # for debug

from flask import Flask, Response, request, jsonify
from g4f.client import Client
import tiktoken  # pip install tiktoken
import time

import json

import traceback

app = Flask(__name__)
client = Client()

MAX_TOKENS_CONTEXT = 1000000  # пока так, по идее, нужно заменить на словарь


import logging
from logging.config import dictConfig

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
            'stream': 'ext://flask.logging.wsgi_errors_stream',  # поток ошибок Flask
            'formatter': 'default'  # использовать формат из formatters.default
        },
        'file': {  # обработчик для записи логов в файл
            'class': 'logging.FileHandler',
            'filename': 'error.log',  # имя файла для логов
            'formatter': 'default',
            'level': 'ERROR',  # записывать только ошибки и выше (ERROR, CRITICAL)
        }
    },
    
    'root': {  # корневой логгер (главный)
        'level': 'INFO',  # минимальный уровень логирования (INFO и выше)
        'handlers': ['wsgi', 'file']  # использовать оба обработчика: в консоль и в файл
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
        total_tokens += 4 + len(enc.encode(msg.get("role", ""))) + len(enc.encode(msg.get("content", "")))
    total_tokens += 2
    return total_tokens

def trim_messages_to_fit(messages, max_tokens=MAX_TOKENS_CONTEXT, model="gpt-4o-mini"):
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
        if len(settings_str) % 4 == 0 and settings_str.replace('-', '').replace('_', '').isalnum():
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

    return {
        "provider": provider_cls,
        "model": model,
        "settings": settings
    }

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
    debug_serialized_data = json.dumps(data, indent=2, ensure_ascii=False)  # for debug, don't delete it
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
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_string,
                "choices": [{
                    "index": 0, # Добавлено согласно спецификации OpenAI
                    "message": {
                        "role": "assistant",
                        "content": response.choices[0].message.content
                    },
                    "finish_reason": response.choices[0].finish_reason if hasattr(response.choices[0], 'finish_reason') else "stop" # Учитываем finish_reason из ответа
                }],
                "usage": response.usage.model_dump() if hasattr(response.usage, "model_dump") else {} # Правильно получаем usage
            })
    except Exception as e:
        # Для отладки можно выводить больше информации об ошибке
        print(f"Error in chat_completions: {e}", exc_info=True)
        app.logger.error(f"Error in chat_completions: {e}\n{traceback.format_exc()}")
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
        completion_tokens = count_tokens([{"role": "assistant", "content": completion}], model_name)

        # Формируем ответ в соответствии со спецификацией Completions API
        result = {
            "id": f"cmpl-{int(time.time())}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": model_string,
            "choices": [
                {
                    "index": 0,
                    "text": completion,
                    "logprobs": None,
                    "finish_reason": response.choices[0].finish_reason if hasattr(response.choices[0], 'finish_reason') else "stop"
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        }

        return jsonify(result)

    except Exception as e:
        print(f"Error in completions: {e}", exc_info=True)
        app.logger.error(f"Error in chat_completions: {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    #app.run(port=3003, debug=True)
    port = int(os.environ.get("PORT", 3003))
    app.run(host="0.0.0.0", port=port)
