import os
from dotenv import load_dotenv
from kiteconnect import KiteConnect

load_dotenv()

API_KEY = os.getenv("KITE_API_KEY")
API_SECRET = os.getenv("KITE_API_SECRET")

if not API_KEY or not API_SECRET:
    raise SystemExit("Missing KITE_API_KEY or KITE_API_SECRET in .env")

kite = KiteConnect(api_key=API_KEY)

print("\n1) Open this URL in your browser and login:")
print(kite.login_url())

print("\n2) After login you’ll be redirected to your Redirect URL.")
print("   Copy ONLY the request_token value from the URL and paste it below.\n")

request_token = input("request_token = ").strip()

data = kite.generate_session(request_token, api_secret=API_SECRET)
access_token = data["access_token"]

print("\n✅ access_token generated:\n", access_token)

# Optional: quick verification
kite.set_access_token(access_token)
profile = kite.profile()
print("\n✅ Connected as:", profile.get("user_name"), "|", profile.get("user_id"))

# Write back into .env (overwrite KITE_ACCESS_TOKEN line)
env_path = ".env"
lines = []
if os.path.exists(env_path):
    with open(env_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

out = []
found = False
for line in lines:
    if line.startswith("KITE_ACCESS_TOKEN="):
        out.append(f"KITE_ACCESS_TOKEN={access_token}")
        found = True
    else:
        out.append(line)

if not found:
    out.append(f"KITE_ACCESS_TOKEN={access_token}")

with open(env_path, "w", encoding="utf-8") as f:
    f.write("\n".join(out) + "\n")

print("\n✅ Saved to .env as KITE_ACCESS_TOKEN")
