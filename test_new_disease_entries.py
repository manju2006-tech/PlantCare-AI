
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DISEASE_FILE = ROOT / "utils" / "disease_info.py"
BACKUP_FILE = ROOT / "utils" / "disease_info_before_new_classes.py"
CLASS_FILE = ROOT / "trained_model_57class_run2" / "class_names.json"

NEW_CLASSES = {
    "Banana___cordana": False,
    "Banana___healthy": True,
    "Banana___pestalotiopsis": False,
    "Banana___sigatoka": False,
    "Money_Plant___Bacterial_Wilt": False,
    "Money_Plant___Healthy": True,
    "Money_Plant___Manganese_Toxicity": False,
    "Rice___Bacterialblight": False,
    "Rice___Blast": False,
    "Rice___Brownspot": False,
    "Rice___Tungro": False,
}

REQUIRED_FIELDS = [
    "plant_name",
    "disease_name",
    "is_healthy",
    "description",
    "causes",
    "treatment",
    "prevention",
    "fertilizer_recommendation",
    "pesticide_recommendation",
]

errors = []


def check(condition, message):
    if not condition:
        errors.append(message)


def main():
    print("===== PlantCare AI Disease Information Test =====")

    if not DISEASE_FILE.exists():
        print("ERROR: utils/disease_info.py was not found.")
        return 1

    if not CLASS_FILE.exists():
        print("ERROR: New model class_names.json was not found.")
        return 1

    # Check duplicate keys in the DISEASE_INFO dictionary source.
    source = DISEASE_FILE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    dictionary_found = False

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name)
            and target.id == "DISEASE_INFO"
            for target in node.targets
        ):
            if isinstance(node.value, ast.Dict):
                keys = [
                    key.value
                    for key in node.value.keys
                    if isinstance(key, ast.Constant)
                    and isinstance(key.value, str)
                ]
                check(
                    len(keys) == len(set(keys)),
                    "Duplicate keys found in DISEASE_INFO."
                )
                dictionary_found = True
                print("Dictionary keys in source:", len(keys))
                break

    check(dictionary_found, "Could not find DISEASE_INFO dictionary.")

    # Import the current information module.
    sys.path.insert(0, str(ROOT))
    from utils.disease_info import DISEASE_INFO, get_disease_info

    print("Entries loaded:", len(DISEASE_INFO))

    # Check all 11 new entries.
    for class_name, expected_healthy in NEW_CLASSES.items():
        entry = DISEASE_INFO.get(class_name)

        if entry is None:
            check(False, f"Missing entry: {class_name}")
            continue

        for field in REQUIRED_FIELDS:
            check(field in entry, f"{class_name}: missing {field}")

        check(
            entry.get("is_healthy") is expected_healthy,
            f"{class_name}: incorrect is_healthy value"
        )

        for field in REQUIRED_FIELDS:
            value = entry.get(field)
            check(
                isinstance(value, str) and bool(value.strip())
                if field != "is_healthy"
                else isinstance(value, bool),
                f"{class_name}: invalid or empty {field}"
            )

        check(
            get_disease_info(class_name) == entry,
            f"{class_name}: lookup returned different information"
        )

    check(
        "Rice___healthy" not in DISEASE_INFO,
        "Rice___healthy should not be added as a model class."
    )

    # Check the actual 57-class model label list.
    classes = json.loads(CLASS_FILE.read_text(encoding="utf-8"))

    check(len(classes) == 57, f"Expected 57 classes, found {len(classes)}")
    check(
        len(classes) == len(set(classes)),
        "Duplicate names found in model class list."
    )

    missing = [name for name in classes if name not in DISEASE_INFO]

    print("Model classes:", len(classes))
    print("Classes with no explicit information:", len(missing))

    for name in missing:
        print("  Generic fallback:", name)

    check(
        all(name in classes for name in NEW_CLASSES),
        "One or more new entries do not match the model's exact class names."
    )

    # Compare original entries against the backup.
    if BACKUP_FILE.exists():
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "backup_disease_info", BACKUP_FILE
        )
        backup_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(backup_module)

        changed = [
            name for name, old_entry in backup_module.DISEASE_INFO.items()
            if DISEASE_INFO.get(name) != old_entry
        ]

        print("Original entries checked:", len(backup_module.DISEASE_INFO))
        check(not changed, f"Original entries changed: {changed}")
    else:
        print("WARNING: Backup not found; old entries were not compared.")

    if errors:
        print("\nFAILED:")
        for error in errors:
            print(" -", error)
        return 1

    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
