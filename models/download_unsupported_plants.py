
import csv
import hashlib
import os
import re
import time
from pathlib import Path

import requests
from PIL import Image
from io import BytesIO

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_ROOT / "dataset" / "unsupported_plant_test"
IMAGE_DIR = OUTPUT_DIR / "images"
METADATA_FILE = OUTPUT_DIR / "metadata.csv"

TARGET_PER_SPECIES = 67
TOTAL_TARGET = 400

API = "https://api.inaturalist.org/v1"
HEADERS = {
    "User-Agent": "PlantCareAI-College-Project/1.0"
}

# Only use photos with these licences.
ALLOWED_LICENSES = {"cc0", "cc-by", "cc-by-sa"}

SPECIES = [
    "Azadirachta indica",       # Neem
    "Rosa",                     # Rose genus
    "Carica papaya",            # Papaya
    "Psidium guajava",          # Guava
    "Cocos nucifera",           # Coconut
    "Hibiscus",   # Hibiscus
]

IMAGE_DIR.mkdir(parents=True, exist_ok=True)

session = requests.Session()
session.headers.update(HEADERS)


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")


def get_json(url, params=None):
    response = session.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def resolve_taxon(scientific_name):
    """Find the exact iNaturalist taxon before downloading observations."""

    data = get_json(
        f"{API}/taxa",
        {
            "q": scientific_name,
            "per_page": 30
        }
    )

    results = data.get("results", [])

    # Prefer an exact scientific-name match.
    exact = [
        taxon for taxon in results
        if taxon.get("name", "").lower() == scientific_name.lower()
    ]

    if not exact:
        raise RuntimeError(
            f"Could not find exact iNaturalist taxon: {scientific_name}"
        )

    # Prefer the expected species/genus rank.
    expected_rank = "genus" if scientific_name in ("Rosa", "Hibiscus") else "species"

    matching_rank = [
        taxon for taxon in exact
        if taxon.get("rank") == expected_rank
    ]

    if not matching_rank:
        raise RuntimeError(
            f"Taxon rank mismatch for {scientific_name}. "
            f"Found ranks: {[t.get('rank') for t in exact]}"
        )

    taxon = matching_rank[0]

    print(
        f"Resolved: {scientific_name} "
        f"(taxon ID: {taxon['id']}, rank: {taxon['rank']})"
    )

    return taxon


def load_existing_metadata():
    rows = []

    if METADATA_FILE.exists():
        with open(METADATA_FILE, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

    return rows


def download_image(url):
    response = session.get(url, timeout=40)
    response.raise_for_status()

    image = Image.open(BytesIO(response.content)).convert("RGB")

    # Reject tiny or likely unusable images.
    if image.width < 150 or image.height < 150:
        raise ValueError("Image resolution too small")

    return image, response.content


def main():
    existing_rows = load_existing_metadata()

    # Resume safely if the script is run again.
    existing_observations = {
        row["observation_id"] for row in existing_rows
    }

    existing_hashes = {
        row["sha256"] for row in existing_rows
    }

    rows = existing_rows[:]

    print(f"\nExisting downloaded images: {len(rows)}")

    for scientific_name in SPECIES:
        species_key = safe_name(scientific_name)

        current_count = sum(
            1 for row in rows
            if row["scientific_name"] == scientific_name
        )

        if current_count >= TARGET_PER_SPECIES:
            print(f"\nAlready complete: {scientific_name}")
            continue

        print(f"\nCollecting: {scientific_name}")
        print(f"Already have: {current_count}/{TARGET_PER_SPECIES}")

        taxon = resolve_taxon(scientific_name)
        taxon_id = taxon["id"]

        page = 1
        downloaded_this_species = 0

        while current_count < TARGET_PER_SPECIES:
            params = {
                "taxon_id": taxon_id,
                "photos": "true",
                "quality_grade": "research",
                "photo_license": "cc0,cc-by,cc-by-sa",
                "per_page": 100,
                "page": page,
                "order_by": "votes",
                "order": "desc"
            }

            data = get_json(
                f"{API}/observations",
                params
            )

            observations = data.get("results", [])

            if not observations:
                print("No more observations available.")
                break

            for observation in observations:
                if current_count >= TARGET_PER_SPECIES:
                    break

                observation_id = str(observation.get("id", ""))

                if not observation_id:
                    continue

                if observation_id in existing_observations:
                    continue

                taxon_info = observation.get("taxon") or {}

                
                photos = observation.get("photos") or []

                if not photos:
                    continue

                photo = photos[0]

                license_code = (
                    photo.get("license_code") or ""
                ).lower()

                if license_code not in ALLOWED_LICENSES:
                    continue

                image_url = photo.get("url")

                if not image_url:
                    continue

                # Request the medium-sized image.
                image_url = image_url.replace(
                    "/square.",
                    "/medium."
                )

                try:
                    image, image_bytes = download_image(image_url)

                    image_hash = hashlib.sha256(
                        image_bytes
                    ).hexdigest()

                    if image_hash in existing_hashes:
                        continue

                    filename = (
                        f"{species_key}_{observation_id}.jpg"
                    )

                    destination = IMAGE_DIR / filename

                    image.save(
                        destination,
                        format="JPEG",
                        quality=92
                    )

                    observer = observation.get("user") or {}

                    row = {
                        "observation_id": observation_id,
                        "scientific_name": scientific_name,
                        "taxon_id": taxon_id,
                        "identified_taxon": taxon_info.get("name", ""),
                        "observer_username": observer.get("login", ""),
                        "photographer_attribution": (
                            photo.get("attribution", "")
                        ),
                        "image_url": image_url,
                        "filename": filename,
                        "license_code": license_code,
                        "observation_url": (
                            f"https://www.inaturalist.org/observations/"
                            f"{observation_id}"
                        ),
                        "sha256": image_hash
                    }

                    rows.append(row)
                    existing_observations.add(observation_id)
                    existing_hashes.add(image_hash)

                    current_count += 1
                    downloaded_this_species += 1

                    print(
                        f"  {scientific_name}: "
                        f"{current_count}/{TARGET_PER_SPECIES}"
                    )

                    # Save metadata after every successful image.
                    with open(
                        METADATA_FILE,
                        "w",
                        newline="",
                        encoding="utf-8"
                    ) as f:
                        writer = csv.DictWriter(
                            f,
                            fieldnames=list(rows[0].keys())
                        )
                        writer.writeheader()
                        writer.writerows(rows)

                except Exception as error:
                    print(
                        f"  Skipped observation {observation_id}: "
                        f"{error}"
                    )

                time.sleep(0.15)

            page += 1

            # Avoid unnecessary API requests.
            if page > 30:
                break

        print(
            f"Finished {scientific_name}: "
            f"{current_count}/{TARGET_PER_SPECIES}"
        )

    print("\n--------------------------------")
    print("DOWNLOAD SUMMARY")
    print("--------------------------------")

    for scientific_name in SPECIES:
        count = sum(
            1 for row in rows
            if row["scientific_name"] == scientific_name
        )
        print(f"{scientific_name}: {count}")

    print(f"\nTotal images: {len(rows)}")
    print(f"Target: {TOTAL_TARGET}")
    print(f"Images: {IMAGE_DIR}")
    print(f"Metadata: {METADATA_FILE}")

    if len(rows) < TOTAL_TARGET:
        print(
            "\nSome species did not reach the target. "
            "Run this script again to resume."
        )


if __name__ == "__main__":
    main()
