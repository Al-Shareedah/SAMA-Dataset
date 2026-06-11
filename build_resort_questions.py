
import csv
import json
import re
from pathlib import Path

# =========================
# EDIT THESE PATHS
# =========================
CATEGORY = "resort"
CSV_FOLDER = Path("resort_csv")
IMAGE_FOLDER = Path("resort_images")
OUTPUT_FILE = Path("resort_questions.json")

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"]


def clean_text(value):
    """Convert missing values to an empty string and trim whitespace."""
    if value is None:
        return ""
    return str(value).strip()


def make_id_text(value):
    """Convert text into an uppercase identifier-safe string."""
    value = value.upper()
    value = re.sub(r"[^A-Z0-9]+", "_", value)
    return value.strip("_")


def find_matching_image(csv_file):
    """
    Find an image with the same filename stem as the CSV.

    Example:
        crystal_mountain_resort.csv
        crystal_mountain_resort.jpg
    """

    # First try an exact filename-stem match.
    for extension in IMAGE_EXTENSIONS:
        candidate = IMAGE_FOLDER / f"{csv_file.stem}{extension}"

        if candidate.exists():
            return candidate

    # Then try a case-insensitive match.
    for image_file in IMAGE_FOLDER.iterdir():
        if (
            image_file.is_file()
            and image_file.suffix.lower() in IMAGE_EXTENSIONS
            and image_file.stem.lower() == csv_file.stem.lower()
        ):
            return image_file

    return None


def get_column(fieldnames, possible_names):
    """Find a CSV column using several accepted header names."""

    normalized_headers = {
        name.strip().lower().replace("_", " "): name
        for name in fieldnames
        if name
    }

    for possible_name in possible_names:
        normalized_name = possible_name.strip().lower().replace("_", " ")

        if normalized_name in normalized_headers:
            return normalized_headers[normalized_name]

    return None


def main():
    if not CSV_FOLDER.exists():
        raise FileNotFoundError(
            f"CSV folder not found: {CSV_FOLDER.resolve()}"
        )

    if not IMAGE_FOLDER.exists():
        raise FileNotFoundError(
            f"Image folder not found: {IMAGE_FOLDER.resolve()}"
        )

    csv_files = sorted(
        CSV_FOLDER.glob("*.csv"),
        key=lambda path: path.name.lower()
    )

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in: {CSV_FOLDER.resolve()}"
        )

    all_questions = []
    seen_question_ids = set()

    for csv_file in csv_files:
        image_file = find_matching_image(csv_file)

        if image_file is None:
            print(
                f"Skipping {csv_file.name}: "
                "no image with the same filename stem was found."
            )
            continue

        category_slug = make_id_text(CATEGORY)
        image_slug = make_id_text(csv_file.stem)

        image_id = f"SAMA_{category_slug}_{image_slug}"

        with csv_file.open(
            "r",
            encoding="utf-8-sig",
            newline=""
        ) as file:
            reader = csv.DictReader(file)

            if not reader.fieldnames:
                print(
                    f"Skipping {csv_file.name}: "
                    "the CSV has no header row."
                )
                continue

            question_column = get_column(
                reader.fieldnames,
                [
                    "Generated Questions",
                    "Generated Question",
                    "Question",
                ],
            )

            generated_answer_column = get_column(
                reader.fieldnames,
                [
                    "Generated Answers",
                    "Generated Answer",
                ],
            )

            true_answer_column = get_column(
                reader.fieldnames,
                [
                    "True Answers",
                    "True Answer",
                    "Ground Truth Answer",
                ],
            )

            output_type_column = get_column(
                reader.fieldnames,
                [
                    "Output_Type",
                    "Output Type",
                    "Map Element Category",
                ],
            )

            required_columns = {
                "question": question_column,
                "generated answer": generated_answer_column,
                "true answer": true_answer_column,
            }

            missing_columns = [
                name
                for name, column in required_columns.items()
                if column is None
            ]

            if missing_columns:
                print(
                    f"Skipping {csv_file.name}: missing columns: "
                    f"{', '.join(missing_columns)}"
                )
                continue

            question_number = 0

            for source_row, row in enumerate(reader, start=2):
                question = clean_text(
                    row.get(question_column)
                )

                generated_answer = clean_text(
                    row.get(generated_answer_column)
                )

                true_answer = clean_text(
                    row.get(true_answer_column)
                )

                if not question:
                    continue

                # Use the manually corrected True Answer when available.
                # Otherwise, use the Generated Answer.
                if true_answer:
                    reference_answer = true_answer
                else:
                    reference_answer = generated_answer

                if not reference_answer:
                    print(
                        f"Skipping {csv_file.name}, "
                        f"row {source_row}: no answer found."
                    )
                    continue

                question_number += 1

                # The image identifier makes the question ID unique
                # across all CSV files.
                question_id = (
                    f"{image_id}_Q{question_number:04d}"
                )

                if question_id in seen_question_ids:
                    raise ValueError(
                        f"Duplicate question ID: {question_id}"
                    )

                seen_question_ids.add(question_id)

                question_record = {
                    "question_id": question_id,
                    "image_id": image_id,
                    "image_filename": image_file.name,
                    "question": question,
                    "reference_answers": [
                        reference_answer
                    ],
                }

                # Keep Output_Type for later accuracy analysis.
                # Do not send this value to the evaluated model.
                if output_type_column is not None:
                    output_type = clean_text(
                        row.get(output_type_column)
                    )

                    if output_type:
                        question_record[
                            "map_element_category"
                        ] = output_type

                all_questions.append(question_record)

        print(
            f"Processed {csv_file.name} -> "
            f"{image_file.name} "
            f"({question_number} questions)"
        )

    output_data = {
        "dataset": "SAMA",
        "category": CATEGORY,
        "question_count": len(all_questions),
        "questions": all_questions,
    }

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            output_data,
            file,
            ensure_ascii=False,
            indent=2
        )

    print()
    print(f"Created: {OUTPUT_FILE.resolve()}")
    print(f"Total questions: {len(all_questions)}")


if __name__ == "__main__":
    main()
