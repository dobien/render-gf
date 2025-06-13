# https://github.com/gpt4free/gpt4free.github.io/blob/main/docs%2Fproviders-and-models.md

import os
# Отключить проверку SSL для всего requests/HF-клиента
#os.environ["CURL_CA_BUNDLE"] = ""
#os.environ["REQUESTS_CA_BUNDLE"] = ""


from flask import Flask, Response, request, jsonify
from g4f.client import Client
import tiktoken  # pip install tiktoken
import time

import json

app = Flask(__name__)
client = Client()

MAX_TOKENS_CONTEXT = 2048

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

# Пример curl для чата:
# curl -X POST http://localhost:3003/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"PollinationsAI:mistral\",\"messages\":[{\"role\":\"user\",\"content\":\"Кто ты?\"}]}"
# curl -X POST http://localhost:3003/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"claude-3.7-sonnet\",\"messages\":[{\"role\":\"user\",\"content\":\"Кто ты?\"}]}"
@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    data = request.get_json()
    model = data.get("model", "gpt-4o-mini")
    messages = data.get("messages", [])
    max_tokens = data.get("max_tokens", MAX_TOKENS_CONTEXT)
    stream = data.get("stream", False)
    # debug_serialized_data = json.dumps(data, indent=2, ensure_ascii=False)  # for debug, don't delete it
    if not isinstance(messages, list) or len(messages) == 0:
        return jsonify({"error": "messages must be a non-empty list"}), 400
    messages = trim_messages_to_fit(messages, MAX_TOKENS_CONTEXT, model)
    try:
        if stream:
            # Создаём генератор для стриминга
            response_generator = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                stream=True,
            )

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
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                stream=False,
            )
            return jsonify({
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
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
        return jsonify({"error": str(e)}), 500



# Пример curl для автодополнения кода:
# curl -X POST http://localhost:3003/v1/completions -H "Content-Type: application/json" -d "{\"model\":\"gpt-4o-mini\",\"prompt\":\"def hello_world():\\n    \",\"max_tokens\":50,\"temperature\":0.5}"

@app.route("/v1/completions", methods=["POST"])
def completions():
    data = request.get_json()
    model = data.get("model", "gpt-4o-mini")
    prompt = data.get("prompt", "")
    max_tokens = data.get("max_tokens", 100)
    temperature = data.get("temperature", 0.7)

    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    try:
        messages = [{"role": "user", "content": prompt}]
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=False,
            temperature=temperature,
            max_tokens=max_tokens
        )
        result = {
            "id": f"cmpl-{int(time.time())}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "text": response.choices[0].message.content,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=3003, debug=True)
