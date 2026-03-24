import argparse
import json
import logging
import os
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def mock_predict(label: str) -> dict:
    return {
        "predicted_food": label.replace("_", " "),
        "confidence": 0.85,
        "notes": "mock_mode"
    }


def gemini_predict(image_path: str, prompt: str) -> dict:
    """
    Gemini API call.
    Requires:
        pip install google-generativeai
        set GEMINI_API_KEY in environment
    """
    import google.generativeai as genai

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables.")

    genai.configure(api_key=api_key)

    model = genai.GenerativeModel("gemini-1.5-flash")

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    response = model.generate_content(
        [
            prompt,
            {
                "mime_type": "image/jpeg",
                "data": image_bytes,
            },
        ]
    )

    text = response.text.strip()

    # Try to parse JSON response
    try:
        parsed = json.loads(text)
    except Exception:
        parsed = {
            "predicted_food": text,
            "confidence": None,
            "notes": "non_json_response"
        }

    return parsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", default="Data-Pipeline/data/processed/food_val.csv", type=str)
    parser.add_argument("--out_jsonl", default="Data-Pipeline/data/processed/food_predictions.jsonl", type=str)
    parser.add_argument("--max_images", default=25, type=int)
    parser.add_argument("--mock", action="store_true", help="Use fake predictions instead of Gemini")
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv).head(args.max_images)

    prompt = """
You are a food image classifier.
Identify the main food shown in the image.
Return ONLY valid JSON in this exact format:
{
  "predicted_food": "food_label_here",
  "confidence": 0.0,
  "notes": "short explanation"
}
Use lowercase text with spaces, not extra commentary.
"""

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            image_path = row["image_path"]
            true_label = row["label"]

            try:
                if args.mock:
                    pred = mock_predict(true_label)
                else:
                    pred = gemini_predict(image_path, prompt)

                record = {
                    "image_path": image_path,
                    "true_label": true_label,
                    "predicted_food": pred.get("predicted_food"),
                    "confidence": pred.get("confidence"),
                    "notes": pred.get("notes", "")
                }

            except Exception as e:
                record = {
                    "image_path": image_path,
                    "true_label": true_label,
                    "predicted_food": None,
                    "confidence": None,
                    "notes": f"error: {str(e)}"
                }

            f.write(json.dumps(record) + "\n")

    logging.info("Saved predictions to %s", out_jsonl)


if __name__ == "__main__":
    main()