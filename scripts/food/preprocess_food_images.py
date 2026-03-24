import argparse
import logging
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

def read_lines(p: Path):
    return [x.strip() for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", default="Data-Pipeline/data/raw/food-101", type=str)
    parser.add_argument("--out_dir", default="Data-Pipeline/data/processed", type=str)
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = raw_root / "meta"
    images = raw_root / "images"

    train_items = read_lines(meta / "train.txt")
    test_items  = read_lines(meta / "test.txt")

    rows = []
    for split, items in [("train", train_items), ("test", test_items)]:
        for item in items:
            label = item.split("/")[0]
            img_path = images / f"{item}.jpg"
            rows.append({"split": split, "label": label, "image_path": str(img_path)})

    df = pd.DataFrame(rows)
    manifest_path = out_dir / "food_manifest.csv"
    df.to_csv(manifest_path, index=False)
    logging.info("Wrote manifest: %s (%d rows)", manifest_path, len(df))

    # Make val split from train
    train_df = df[df["split"] == "train"].copy()
    train_df, val_df = train_test_split(
        train_df, test_size=0.2, random_state=42, stratify=train_df["label"]
    )
    train_out = out_dir / "food_train.csv"
    val_out = out_dir / "food_val.csv"
    train_df.to_csv(train_out, index=False)
    val_df.to_csv(val_out, index=False)
    logging.info("Wrote train/val: %s (%d), %s (%d)", train_out, len(train_df), val_out, len(val_df))

    # Class distribution (simple bias check)
    dist = train_df["label"].value_counts().reset_index()
    dist.columns = ["label", "count"]
    dist_out = out_dir / "food_class_distribution.csv"
    dist.to_csv(dist_out, index=False)
    logging.info("Wrote class distribution: %s", dist_out)

if __name__ == "__main__":
    main()