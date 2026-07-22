import httpx

# Curated Wikimedia Commons URLs - manually verified to exist and be relevant
WIKIMEDIA_URLS = {
    "Adidas Stan Smith":     "https://upload.wikimedia.org/wikipedia/commons/2/28/Adidas_Stan_Smiths_sneaker.jpg",
    "Adidas Superstar":      "https://upload.wikimedia.org/wikipedia/commons/f/f2/Adidas_Superstar_%28White_and_Black%29.jpg",
    "Adidas Samba":          "https://upload.wikimedia.org/wikipedia/commons/0/0c/Adidas_Samba.jpg",
    "Adidas Ultraboost":     "https://upload.wikimedia.org/wikipedia/commons/4/49/Adidas_Running_Shoe_Demo.jpg",
    "Nike Dunk":             "https://upload.wikimedia.org/wikipedia/commons/9/98/Nike_SB_Dunk_High_MF_Doom_edited.jpg",
    "Nike Air Force 1":      "https://upload.wikimedia.org/wikipedia/commons/b/b5/Nike-Air-Force-1-Low-2007.jpg",
    "Converse Chuck 70":     "https://upload.wikimedia.org/wikipedia/commons/9/99/Converse-Chuck_70.jpg",
    "Converse Chuck Taylor": "https://upload.wikimedia.org/wikipedia/commons/3/34/Converse_All_Star_Taylor_Chuck.jpg",
    "Reebok Classic":        "https://upload.wikimedia.org/wikipedia/commons/4/45/Reebok_Classic_Leather.jpg",
    "Vans Old Skool":        "https://upload.wikimedia.org/wikipedia/commons/3/3f/Vans_Old_Skool_sneakers_%282%29.jpg",
    "Asics Gel Running":     "https://upload.wikimedia.org/wikipedia/commons/5/5e/ASICS_running_shoe.jpg",
    "Fila Disruptor":        "https://upload.wikimedia.org/wikipedia/commons/7/74/Fila_Disruptor_2.jpg",
    "Skechers":              "https://upload.wikimedia.org/wikipedia/commons/c/c4/Skechers_shoes.jpg",
    "Nike Air Max":          "https://upload.wikimedia.org/wikipedia/commons/0/00/Nike_Air_Max_90.jpg",
    "Nike Blazer":           "https://upload.wikimedia.org/wikipedia/commons/e/e5/Nike_Blazer_Mid.jpg",
    "Adidas Gazelle":        "https://upload.wikimedia.org/wikipedia/commons/1/14/Adidas_Gazelle_sneaker.jpg",
}

for name, url in WIKIMEDIA_URLS.items():
    try:
        resp = httpx.get(url, headers={"User-Agent": "SneakVault/1.0"}, timeout=10, follow_redirects=True)
        ct = resp.headers.get('content-type', '')
        size = len(resp.content)
        if 'image' in ct and size > 2000:
            print(f"  OK  ({size//1024:>4}KB): {name}")
        else:
            print(f"  BAD ({resp.status_code}, {ct[:25]}, {size}B): {name}")
    except Exception as e:
        print(f"  FAIL ({type(e).__name__}): {name}")
