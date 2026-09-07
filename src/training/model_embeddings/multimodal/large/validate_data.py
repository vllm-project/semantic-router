#!/usr/bin/env python3
import argparse
import os

from .data import JsonlManifestDataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate multimodal training manifest"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--audio-root", default=None)
    parser.add_argument("--require-negative", action="store_true")
    parser.add_argument("--check-files", action="store_true")
    args = parser.parse_args()

    ds = JsonlManifestDataset(
        manifest_path=args.manifest,
        image_root=args.image_root,
        audio_root=args.audio_root,
        allow_missing_negative=not args.require_negative,
    )

    missing = []
    for idx in range(len(ds)):
        rec = ds[idx]
        items = [rec.query, rec.positive]
        if rec.negative is not None:
            items.append(rec.negative)
        for item in items:
            if (
                item.modality in {"image", "audio"}
                and args.check_files
                and not os.path.exists(item.value)
            ):
                missing.append((idx, item.modality, item.value))

    print(f"Loaded {len(ds)} records from {args.manifest}")
    if missing:
        print(f"Missing media files: {len(missing)}")
        for idx, modality, path in missing[:20]:
            print(f"  line={idx} modality={modality} path={path}")
        raise SystemExit(1)

    print("Validation passed")


if __name__ == "__main__":
    main()
