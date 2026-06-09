import os
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

load_dotenv()

LARGE = os.getenv("LARGE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
SMALL = os.getenv("SMALL_MODEL", "Qwen/Qwen2.5-3B-Instruct")

for model_id in dict.fromkeys([LARGE, SMALL]):
    print(f"Checking {model_id} ...")
    snapshot_download(repo_id=model_id)
    print(f"  {model_id} ready.")

print("All models ready.")
