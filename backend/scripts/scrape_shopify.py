"""
SneakVault — Real Sneaker Scraper (VegNonVeg + Superkicks)
Pulls real product data from Shopify JSON endpoints.
No API key needed, no JS rendering, no auth.
"""

import sys
import os

# Fix Windows console encoding for Unicode
if sys.platform == 'win32':
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import json
import time
import urllib.request
import urllib.error
import re
import hashlib
from pathlib import Path
from PIL import Image
import io

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "app" / "data"
IMAGES_DIR = BASE_DIR / "app" / "assets" / "images"

MAX_SNEAKERS_PER_SOURCE = 1000  # Cap to keep dataset manageable

SOURCES = {
    "vegnonveg": {
        "base_url": "https://www.vegnonveg.com",
        "products_url": "https://www.vegnonveg.com/collections/all/products.json",
        "prefix": "vnv",
        "label": "VegNonVeg",
    },
    "superkicks": {
        "base_url": "https://www.superkicks.in",
        "products_url": "https://www.superkicks.in/products.json",
        "prefix": "sk",
        "label": "Superkicks",
    },
}

# Keywords to SKIP (non-sneaker products)
SKIP_TYPES = {
    "slide", "slides", "sandal", "sandals", "slipper", "slippers",
    "flip flop", "flip-flop", "clog", "clogs", "mule", "mules",
    "sock", "socks", "apparel", "clothing", "t-shirt", "tee",
    "hoodie", "jacket", "shorts", "pants", "jogger", "cap", "hat",
    "bag", "backpack", "accessories", "keychain", "lace", "laces",
    "insole", "cleaner", "care", "poster", "sticker", "gift card",
}

# Keywords that confirm it's a sneaker
SNEAKER_KEYWORDS = {
    "sneaker", "sneakers", "shoe", "shoes", "trainer", "trainers",
    "runner", "running", "basketball", "skate", "low", "mid", "high",
    "retro", "dunk", "jordan", "air max", "air force", "yeezy",
    "forum", "samba", "gazelle", "old skool", "chuck", "gel",
    "ultraboost", "nmd", "550", "574", "990", "2002",
}

HEADERS = {
    "User-Agent": "SneakVault/2.0 (educational sneaker discovery project)",
    "Accept": "application/json",
}

REQUEST_DELAY = 1.5  # seconds between requests


def fetch_json(url):
    """Fetch JSON from a URL with proper error handling."""
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        print(f"  HTTP {e.code}: {url}")
        return None
    except Exception as e:
        print(f"  Error fetching {url}: {e}")
        return None


def is_sneaker(product):
    """Determine if a Shopify product is a sneaker (not slides/apparel/accessories)."""
    title = product.get("title", "").lower()
    ptype = product.get("product_type", "").lower()
    tags = " ".join(product.get("tags", [])).lower() if isinstance(product.get("tags"), list) else str(product.get("tags", "")).lower()

    combined = f"{title} {ptype} {tags}"

    # Hard skip if matches non-sneaker keywords
    for skip in SKIP_TYPES:
        if skip in combined:
            return False

    # Accept if product_type or tags mention sneaker-related terms
    for kw in SNEAKER_KEYWORDS:
        if kw in combined:
            return True

    # Default: accept if product_type contains "footwear" or "shoe"
    if "footwear" in ptype or "shoe" in ptype:
        return True

    # If we can't tell, skip to be safe
    return False


def extract_brand(product):
    """Extract brand from Shopify product vendor field."""
    vendor = product.get("vendor", "").strip()
    if vendor:
        return vendor
    # Fallback: try to parse from title
    title = product.get("title", "")
    known_brands = [
        "Nike", "Adidas", "Puma", "New Balance", "Reebok", "Converse",
        "Vans", "Asics", "Skechers", "Fila", "Jordan", "Under Armour",
    ]
    for brand in known_brands:
        if brand.lower() in title.lower():
            return brand
    return "Unknown"


def extract_colorway(product):
    """Try to extract colorway from title or variant."""
    title = product.get("title", "")
    # Many Shopify sneaker titles have format: "Brand Model - Colorway"
    for sep in [" - ", " – ", " — ", " | "]:
        if sep in title:
            return title.split(sep)[-1].strip()

    # Try variant title
    variants = product.get("variants", [])
    if variants:
        v_title = variants[0].get("title", "")
        if v_title and v_title != "Default Title":
            return v_title

    return "Standard"


def extract_price(product):
    """Get price in INR (Shopify stores price as string with decimals)."""
    variants = product.get("variants", [])
    if variants:
        price_str = variants[0].get("price", "0")
        try:
            price = float(price_str)
            # Shopify Indian stores typically store in rupees already
            return int(price)
        except (ValueError, TypeError):
            return 0
    return 0


def extract_image_url(product):
    """Get the best product image URL."""
    images = product.get("images", [])
    if images:
        src = images[0].get("src", "")
        # Shopify CDN — request 600px wide version
        if src and "cdn.shopify.com" in src:
            # Strip any existing size suffix and add our own
            src = re.sub(r'_\d+x\d*\.', '_600x.', src)
            if '_600x.' not in src:
                # Add size before extension
                src = re.sub(r'\.(\w+)$', r'_600x.\1', src)
        return src
    return ""


def clean_description(body_html):
    """Strip HTML tags from Shopify body_html."""
    if not body_html:
        return ""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', body_html)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Truncate to 200 chars
    if len(text) > 200:
        text = text[:197] + "..."
    return text


def download_image(url, filepath):
    """Download an image and save as normalized JPEG."""
    req = urllib.request.Request(url, headers={
        "User-Agent": HEADERS["User-Agent"],
        "Accept": "image/*",
    })
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = resp.read()
        img = Image.open(io.BytesIO(data))
        img.convert("RGB").save(filepath, "JPEG", quality=88)
        return True
    except Exception as e:
        print(f"    Image download failed: {e}")
        return False


def scrape_source(source_key, source_config, existing_names):
    """Scrape all sneakers from one Shopify source."""
    prefix = source_config["prefix"]
    label = source_config["label"]
    base_url = source_config["base_url"]
    products_url = source_config["products_url"]

    print(f"\n{'='*60}")
    print(f"  Scraping {label} ({base_url})")
    print(f"{'='*60}")

    all_products = []
    page = 1
    max_pages = 20  # Safety limit (~1000 products max)

    while page <= max_pages:
        url = f"{products_url}?page={page}&limit=250"
        print(f"  Fetching page {page}...")
        data = fetch_json(url)

        if not data or "products" not in data:
            print(f"  No data on page {page}, stopping.")
            break

        products = data["products"]
        if not products:
            print(f"  Empty page {page}, done.")
            break

        all_products.extend(products)
        print(f"  Got {len(products)} products (total: {len(all_products)})")

        page += 1
        time.sleep(REQUEST_DELAY)

    print(f"\n  Total raw products from {label}: {len(all_products)}")

    # Filter to sneakers only
    sneakers = [p for p in all_products if is_sneaker(p)]
    print(f"  After sneaker filter: {len(sneakers)}")

    # Process into our schema
    processed = []
    img_failed = 0

    for i, product in enumerate(sneakers):
        name = product.get("title", "").strip()
        brand = extract_brand(product)
        price = extract_price(product)

        # Skip duplicates (same name already scraped from other source)
        name_key = re.sub(r'[^a-z0-9]', '', name.lower())
        if name_key in existing_names:
            continue
        existing_names.add(name_key)

        # Skip if no price or unreasonable price
        if price < 500 or price > 100000:
            continue

        snk_id = f"snk-{prefix}{i+1:03d}"
        filename = f"{snk_id}.jpg"
        filepath = IMAGES_DIR / filename
        image_url = extract_image_url(product)
        handle = product.get("handle", "")
        source_url = f"{base_url}/products/{handle}" if handle else ""

        # Download image
        if image_url:
            ok = download_image(image_url, filepath)
            if not ok:
                img_failed += 1
                continue  # Skip products where image download fails
        else:
            continue  # Skip products with no image

        entry = {
            "id": snk_id,
            "name": name,
            "brand": brand,
            "price": price,
            "colorway": extract_colorway(product),
            "description": clean_description(product.get("body_html", "")),
            "isTrending": False,
            "image_filename": filename,
            "source": source_key,
            "source_url": source_url,
        }
        processed.append(entry)

        status = f"[{len(processed):03d}] {brand} - {name[:50]} - Rs.{price}"
        print(f"  {status}")

        # Brief pause every 10 images
        if len(processed) % 10 == 0:
            time.sleep(0.5)

        # Cap per source
        if len(processed) >= MAX_SNEAKERS_PER_SOURCE:
            print(f"  Reached {MAX_SNEAKERS_PER_SOURCE} cap, stopping.")
            break

    print(f"\n  {label} result: {len(processed)} sneakers, {img_failed} image failures")
    return processed


def scrape_all(dry_run=False):
    """Main scrape function — fetches from all sources."""
    print("=" * 60)
    print("  SneakVault — Real Sneaker Scraper")
    print("=" * 60)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    all_sneakers = []
    existing_names = set()

    for key, config in SOURCES.items():
        sneakers = scrape_source(key, config, existing_names)
        all_sneakers.extend(sneakers)

    # Sort: trending first, then by brand, then by price
    all_sneakers.sort(key=lambda s: (s["brand"], s["price"]))

    # Save scraped data
    scraped_file = DATA_DIR / "scraped_sneakers.json"
    with open(scraped_file, "w", encoding="utf-8") as f:
        json.dump(all_sneakers, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  SCRAPING COMPLETE")
    print(f"  Total: {len(all_sneakers)} real sneakers")
    print(f"  Saved to: {scraped_file}")
    print(f"{'='*60}")

    # Print brand breakdown
    brands = {}
    for s in all_sneakers:
        brands[s["brand"]] = brands.get(s["brand"], 0) + 1
    print(f"\n  Brand breakdown:")
    for brand, count in sorted(brands.items(), key=lambda x: -x[1]):
        print(f"    {brand}: {count}")

    # Print price range breakdown
    under_5k = sum(1 for s in all_sneakers if s["price"] < 5000)
    mid = sum(1 for s in all_sneakers if 5000 <= s["price"] < 10000)
    over_10k = sum(1 for s in all_sneakers if s["price"] >= 10000)
    print(f"\n  Price breakdown:")
    print(f"    Under Rs.5,000: {under_5k}")
    print(f"    Rs.5,000-Rs.10,000: {mid}")
    print(f"    Over Rs.10,000: {over_10k}")

    return all_sneakers


if __name__ == "__main__":
    import sys
    dry_run = "--dry-run" in sys.argv
    scrape_all(dry_run=dry_run)
