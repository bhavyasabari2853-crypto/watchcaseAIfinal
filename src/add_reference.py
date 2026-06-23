import argparse
import os
import sys

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

from src.reference_manager import ReferenceManager


def main():
    parser = argparse.ArgumentParser(description="Add a reference watch-case image")
    parser.add_argument("image", help="Path to the image file")
    parser.add_argument("name", help="Name for the reference (e.g. Case_20)")
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print(f"Error: File not found: {args.image}")
        sys.exit(1)

    mgr = ReferenceManager(ref_dir=os.path.join(_BASE, "references"))
    mgr.add(args.image, args.name)


if __name__ == "__main__":
    main()
