import urllib.request
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
folder_id = "1wZa2kZkUSTKE9LFCijYBUsFhXRP7ONLw"
url = f"https://drive.google.com/drive/folders/{folder_id}"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

with urllib.request.urlopen(req) as resp:
    content = resp.read().decode("utf-8", errors="ignore")

pos = content.find("٢٠٢٦١١٠١_٢١٤٩١٠.mp4")
if pos != -1:
    snippet = content[max(0, pos-500):min(len(content), pos+500)]
    print("Snippet:")
    print(snippet)
    ids = re.findall(r'["\']([a-zA-Z0-9_-]{28,45})["\']', snippet)
    print("\nPossible IDs:", ids)
else:
    print("Not found directly by name")

