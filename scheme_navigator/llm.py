
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:7b-instruct"



def ask_llm(system_prompt, user_prompt, temperature=0.2):
    import time

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "stream": False,
        "options": {
            "temperature": temperature
        },
        "keep_alive": "10m"
    }

    start = time.time()

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=180
    )
    response.raise_for_status()

    data = response.json()

    elapsed = time.time() - start

    print(
        f"[LLM TIMING] Total: {elapsed:.2f}s | "
        f"Load: {data.get('load_duration', 0) / 1e9:.2f}s | "
        f"Prompt: {data.get('prompt_eval_duration', 0) / 1e9:.2f}s | "
        f"Generation: {data.get('eval_duration', 0) / 1e9:.2f}s | "
        f"Tokens: {data.get('eval_count', 0)}",
        flush=True
    )

    return data["message"]["content"]

if __name__ == "__main__":
    print("Testing Ollama...")

    answer = ask_llm(
        "You are a helpful assistant.",
        "Explain government schemes in simple English."
    )

    print("Qwen response:")
    print(answer)
