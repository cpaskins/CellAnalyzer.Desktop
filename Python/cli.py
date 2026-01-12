import argparse
import json
import os
import cv2

from masterFunction import cell_detection
from parameters import default_parameters


def main():
    parser = argparse.ArgumentParser(description="CellAnalyzer CLI")

    # Defaults mode
    parser.add_argument("--defaults", action="store_true",
                        help="Print default parameters as JSON and exit")

    # Normal mode args (NOT required here)
    parser.add_argument("--image", help="Path to input image")
    parser.add_argument("--output", help="Path to output JSON (result.json)")
    parser.add_argument("--params", help="Path to params JSON")

    args = parser.parse_args()

    # --- Defaults mode ---
    if args.defaults:
        print(json.dumps(default_parameters(), indent=2))
        return

    # --- Normal analysis mode ---
    if not args.image or not args.output:
        parser.error("--image and --output are required unless --defaults is used")

    image_path = args.image
    output_json = args.output
    output_dir = os.path.dirname(output_json)
    os.makedirs(output_dir, exist_ok=True)

    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(f"Failed to load image: {image_path}")

    if args.params:
        with open(args.params, "r", encoding="utf-8") as f:
            params = json.load(f)
    else:
        params = default_parameters()

    result = cell_detection(image, **params)

    overlay_path = os.path.join(output_dir, "overlay.png")
    cv2.imwrite(overlay_path, result["images"]["overlay"])

    mask_path = os.path.join(output_dir, "mask.png")
    cv2.imwrite(mask_path, result["images"]["mask"])

    json_result = {
        "counts": result["counts"],
        "areas": result["areas"],
        "fluorescence": result["fluorescence"],
        "images": {"overlay": "overlay.png", "mask": "mask.png"}
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(json_result, f, indent=2)

    print("OK")


if __name__ == "__main__":
    main()
