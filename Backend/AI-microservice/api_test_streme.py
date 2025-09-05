import requests

url = "http://127.0.0.1:8000/chat/stream"   # <-- FIXED

payload = {
    "user_input": "ok, if i wanted to be a backend developer what should i learn or study?",
    "role": "Career_mentor"
}

with requests.post(url, json=payload, stream=True) as res:
    if res.status_code == 200:
        print("✅ Streaming response:")
        for chunk in res.iter_content(chunk_size=None, decode_unicode=True):
            if chunk:
                print(chunk, end="", flush=True)
    else:
        print("❌ Error:", res.status_code, res.text)

#python C:\Users\Lenovo\Documents\programing\miniProject\Backend\AI-microservice\api_test_streme.py
