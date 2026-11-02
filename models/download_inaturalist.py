
import csv
import hashlib
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests
from PIL import Image
from io import BytesIO


# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "plant_binary_source"
    / "inaturalist_download"
)

IMAGE_DIR = OUTPUT_DIR / "images"
METADATA_FILE = OUTPUT_DIR / "metadata.csv"

API_URL = "https://api.inaturalist.org/v1/observations"

TARGET_IMAGES = 1000
PER_PAGE = 100
REQUEST_DELAY = 1.1

# Only photos with this exact licence are accepted.
ALLOWED_LICENSES = {"cc-by"}

ALLOWED_HOSTS = {
    "static.inaturalist.org",
    "inaturalist-open-data.s3.amazonaws.com",
}

CSV_FIELDS = [
    "observation_id",
    "photo_id",
    "scientific_name",
    "observer_username",
    "photographer_attribution",
    "image_url",
    "filename",
    "license_code",
    "license_url",
    "observation_url",
    "observed_on",
    "downloaded_at",
    "sha256",
]

USER_AGENT = "PlantCareAI-CollegeProject/1.0 (educational dataset collection)"


# --------------------------------------------------
# HTTP SESSION AND RATE LIMITING
# --------------------------------------------------

session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})

last_request_time = 0.0


def safe_get(url, params=None, stream=False):
    """Make a rate-limited request with basic retry handling."""

    global last_request_time

    for attempt in range(4):
        elapsed = time.monotonic() - last_request_time

        if elapsed < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - elapsed)

        try:
            response = session.get(
                url,
                params=params,
                timeout=60,
                stream=stream,
            )
            last_request_time = time.monotonic()

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")

                try:
                    wait = float(retry_after)
                except (TypeError, ValueError):
                    wait = 10 * (attempt + 1)

                print(f"Rate limited. Waiting {wait:.1f} seconds...")
                response.close()
                time.sleep(wait)
                continue

            if response.status_code >= 500:
                response.close()
                time.sleep(3 * (attempt + 1))
                continue

            response.raise_for_status()
            return response

        except requests.RequestException as error:
            print(f"Request error: {error}")

            if attempt == 3:
                return None

            time.sleep(3 * (attempt + 1))

    return None


# --------------------------------------------------
# LICENCE INFORMATION
# --------------------------------------------------

def get_license_url(license_code):
    if license_code == "cc-by":
        return "https://creativecommons.org/licenses/by/4.0/"

    return ""


# --------------------------------------------------
# IMAGE URL VALIDATION
# --------------------------------------------------

def get_medium_url(original_url):
    """Request the medium-sized iNaturalist photo."""

    if not original_url:
        return None

    parsed = urlparse(original_url)

    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS:
        return None

    return re.sub(
        r"/(square|small|thumb|large|original)\.",
        "/medium.",
        original_url,
    )


# --------------------------------------------------
# EXISTING DOWNLOAD TRACKING
# --------------------------------------------------

def load_existing_records():
    photo_ids = set()
    image_hashes = set()

    if not METADATA_FILE.exists():
        return photo_ids, image_hashes

    with METADATA_FILE.open(
        "r", newline="", encoding="utf-8"
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            photo_ids.add(row["photo_id"])

            if row.get("sha256"):
                image_hashes.add(row["sha256"])

    return photo_ids, image_hashes


def initialize_metadata():
    if not METADATA_FILE.exists():
        with METADATA_FILE.open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
            writer.writeheader()


# --------------------------------------------------
# IMAGE DOWNLOAD AND VALIDATION
# --------------------------------------------------

def download_photo(observation, photo, existing_ids, existing_hashes):

    photo_id = str(photo.get("id", ""))

    if not photo_id or photo_id in existing_ids:
        return False

    license_code = (photo.get("license_code") or "").lower()

    if license_code not in ALLOWED_LICENSES:
        return False

    image_url = get_medium_url(photo.get("url"))

    if not image_url:
        print(f"Skipping photo {photo_id}: invalid image URL")
        return False

    response = safe_get(image_url, stream=True)

    if response is None:
        return False

    try:
        content_type = response.headers.get(
            "Content-Type", ""
        ).lower()

        if not content_type.startswith("image/"):
            print(f"Skipping photo {photo_id}: not an image")
            return False

        image_data = response.content

    finally:
        response.close()

    if not image_data:
        return False

    image_hash = hashlib.sha256(image_data).hexdigest()

    if image_hash in existing_hashes:
        print(f"Skipping duplicate image: {photo_id}")
        existing_ids.add(photo_id)
        return False

    try:
        image = Image.open(BytesIO(image_data))
        image.verify()

        # Reopen after verify to obtain the actual image format.
        image = Image.open(BytesIO(image_data))
        image_format = image.format

        extensions = {
            "JPEG": ".jpg",
            "PNG": ".png",
            "WEBP": ".webp",
        }

        extension = extensions.get(image_format)

        if extension is None:
            print(f"Skipping unsupported format: {photo_id}")
            return False

    except Exception as error:
        print(f"Invalid image {photo_id}: {error}")
        return False

    filename = f"obs_{observation['id']}_photo_{photo_id}{extension}"
    image_path = IMAGE_DIR / filename

    # Do not overwrite an existing file.
    if image_path.exists():
        print(f"File already exists: {filename}")
        existing_ids.add(photo_id)
        return False

    image_path.write_bytes(image_data)

    taxon = observation.get("taxon") or {}
    observer = observation.get("user") or {}

    metadata = {
        "observation_id": observation["id"],
        "photo_id": photo_id,
        "scientific_name": taxon.get("name", ""),
        "observer_username": observer.get("login", ""),
        "photographer_attribution": photo.get("attribution", ""),
        "image_url": image_url,
        "filename": filename,
        "license_code": license_code,
        "license_url": get_license_url(license_code),
        "observation_url": (
            f"https://www.inaturalist.org/observations/"
            f"{observation['id']}"
        ),
        "observed_on": observation.get("observed_on", ""),
        "downloaded_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "sha256": image_hash,
    }

    with METADATA_FILE.open(
        "a", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writerow(metadata)
        file.flush()

    existing_ids.add(photo_id)
    existing_hashes.add(image_hash)

    return True


# --------------------------------------------------
# MAIN DOWNLOADER
# --------------------------------------------------

def main():

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    initialize_metadata()

    existing_ids, existing_hashes = load_existing_records()

    downloaded = len(existing_ids)

    print("=" * 55)
    print("PlantCare AI - iNaturalist Image Downloader")
    print("=" * 55)
    print(f"Target images: {TARGET_IMAGES}")
    print(f"Already recorded: {downloaded}")

    if downloaded >= TARGET_IMAGES:
        print("Target already reached. Nothing to download.")
        return

    page = 1
    failed_pages = 0

    while downloaded < TARGET_IMAGES:

        params = {
            "iconic_taxa[]": "Plantae",
            "quality_grade": "research",
            "has[]": "photos",
            "photo_license": "CC-BY",
            "per_page": PER_PAGE,
            "page": page,
            "order_by": "observed_on",
            "order": "desc",
        }

        print(f"\nFetching observation page {page}...")

        response = safe_get(API_URL, params=params)

        if response is None:
            failed_pages += 1

            if failed_pages >= 3:
                print("Stopping after repeated API failures.")
                break

            page += 1
            continue

        failed_pages = 0

        try:
            observations = response.json().get("results", [])
        except ValueError:
            print("Invalid API response. Stopping.")
            response.close()
            break

        response.close()

        if not observations:
            print("No more observations available.")
            break

        for observation in observations:

            photos = observation.get("photos") or []

            for photo in photos:

                if downloaded >= TARGET_IMAGES:
                    break

                success = download_photo(
                    observation,
                    photo,
                    existing_ids,
                    existing_hashes,
                )

                if success:
                    downloaded += 1

                    print(
                        f"[{downloaded}/{TARGET_IMAGES}] "
                        f"Downloaded photo {photo.get('id')}"
                    )

            if downloaded >= TARGET_IMAGES:
                break

        page += 1

        # iNaturalist pagination is limited.
        if page > 100:
            print("Reached the 10,000-observation pagination limit.")
            break

    print("\n" + "=" * 55)
    print("DOWNLOAD SUMMARY")
    print("=" * 55)
    print(f"Images recorded: {downloaded}")
    print(f"Image folder: {IMAGE_DIR}")
    print(f"Metadata file: {METADATA_FILE}")
    print("Original 57-class dataset was not modified.")


if __name__ == "__main__":
    main()
