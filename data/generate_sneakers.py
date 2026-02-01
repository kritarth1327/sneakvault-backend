import json
import random

BRANDS = [
    "Nike", "Adidas", "Puma", "Reebok", "New Balance",
    "Asics", "Converse", "Vans", "Jordan", "Yeezy"
]

MODELS = [
    "Air Max", "Ultraboost", "Classic", "Retro", "Runner",
    "Pro", "Elite", "Sport", "Boost", "Zoom"
]

def generate_price():
    tier = random.choice(["budget", "mid", "heat", "premium"])

    if tier == "budget":
        return random.randint(1500, 4999)
    elif tier == "mid":
        return random.randint(5000, 9999)
    elif tier == "heat":
        return random.randint(10000, 24999)
    else:
        return random.randint(25000, 90000)

def make_item(i):
    brand = random.choice(BRANDS)
    model = random.choice(MODELS)
    price_inr = generate_price()

    return {
        "id": i,
        "brand": brand,
        "model": f"{brand} {model} {i}",
        "price_inr": price_inr,
        "image": f"https://via.placeholder.com/300?text={brand}+{model}"
    }

def generate(n=200):
    return [make_item(i) for i in range(n)]

if __name__ == "__main__":
    sneakers = generate(200)
    with open("backend/data/sneakers.json", "w") as f:
        json.dump(sneakers, f, indent=2)
    print("Saved 200 updated sneakers to backend/data/sneakers.json")
