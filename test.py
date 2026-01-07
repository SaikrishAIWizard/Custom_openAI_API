import requests
import json

API_URL = "https://custom-openai-api.onrender.com/format-product"
# If deployed, example:
# API_URL = "https://your-app-name.onrender.com/format-product"

payload = {
    "text": """Brand - MIX

PURE COTTON✌️
PROPER BRAND ACCESSORIES✌️
Fully OG Fabric🔥

QUALITY PRODUCT
SMART LOOK🥰

SIZE-M38 L40 XL42 XXL44
(BRAND STANDARD SIZES)

AWESOME 2 COLORS😍

PRICE - 399/-(FREE SHIP)💪

BEST QUALITY BEST PRICE

Take Open Orders

https://i.ibb.co/hFL5GrDF/Whats-App-Image-2026-01-07-at-00-05-06.jpg
https://i.ibb.co/KzVkj1vn/Whats-App-Image-2026-01-07-at-00-05-05.jpg
https://i.ibb.co/BHW8Lg9z/Whats-App-Image-2026-01-07-at-00-05-04.jpg
"""
}

headers = {
    "Content-Type": "application/json"
}

try:
    response = requests.post(
        API_URL,
        headers=headers,
        data=json.dumps(payload),
        timeout=120
    )

    print("Status Code:", response.status_code)

    if response.status_code == 200:
        print("\nFormatted Output:\n")
        print(response.json()["formatted_text"])
    else:
        print("\nError Response:")
        print(response.text)

except requests.exceptions.RequestException as e:
    print("Request failed:", str(e))
