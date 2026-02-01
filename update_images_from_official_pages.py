import json
import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

CDN_KEYWORDS = [
    "static.nike.com",
    "assets.adidas.com",
    "images.puma.com",
    "nb.scene7.com"
]

def extract_best_image(product_url):
    res = requests.get(product_url, headers=HEADERS, timeout=10)
    soup = BeautifulSoup(res.text, "html.parser")

    images = []

    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src")
        if not src:
            continue

        if any(k in src for k in CDN_KEYWORDS):
            if any(bad in src.lower() for bad in ["logo", "icon", "sprite"]):
                continue
            images.append(src)

    # Heuristic: longest URL = usually highest-res product image
    images.sort(key=len, reverse=True)
    return images[0] if images else None


def main():
    with open("backend/data/sneakers.json", "r", encoding="utf-8") as f:
        sneakers = json.load(f)

    updated = 0

    for s in sneakers:
        if s.get("product_url") and not s.get("image"):
            print(f"→ Updating {s['brand']} {s['model']}")

            img = extract_best_image(s["product_url"])
            if img:
                s["image"] = img
                updated += 1
                print("   ✅ Image set")
            else:
                print("   ❌ No image found")

    with open("backend/data/sneakers.json", "w", encoding="utf-8") as f:
        json.dump(sneakers, f, indent=2)

    print(f"\n🎉 Done — updated {updated} sneakers")


if __name__ == "__main__":
    main()
