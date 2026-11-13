import argparse
import sys
from pathlib import Path

import pandas as pd
from neo4j import GraphDatabase

sys.path.append(str(Path.cwd()))
from backend.repositories.csv_repository import ALIASES


def normalize_value(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default="../data/bns_sections.csv",
    )
    parser.add_argument(
        "--uri",
        default="bolt://localhost:7687",
    )
    parser.add_argument(
        "--user",
        default="neo4j",
    )
    parser.add_argument(
        "--password",
        required=True,
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # 1. Load dataset
    # ---------------------------------------------------------

    frame = pd.read_csv(
        args.data,
        dtype=str,
    )

    print(f"Loaded {len(frame)} rows")
    print("Original columns:")
    print(frame.columns.tolist())

    # ---------------------------------------------------------
    # 2. Normalize column names
    # ---------------------------------------------------------

    rename_map = {}

    for target, candidates in ALIASES.items():
        source = next(
            (column for column in candidates if column in frame.columns),
            None,
        )

        if source:
            rename_map[source] = target

    frame = frame.rename(columns=rename_map)

    print("\nNormalized columns:")
    print(frame.columns.tolist())

    # ---------------------------------------------------------
    # 3. Validate required columns
    # ---------------------------------------------------------

    required_columns = [
        "chapter_number",
        "chapter_name",
        "section_number",
        "section_name",
        "description",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in frame.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Optional field
    if "chapter_subtype" not in frame.columns:
        frame["chapter_subtype"] = ""

    # ---------------------------------------------------------
    # 4. Normalize values
    # ---------------------------------------------------------

    columns_to_clean = [
        "chapter_number",
        "chapter_name",
        "chapter_subtype",
        "section_number",
        "section_name",
        "description",
    ]

    for column in columns_to_clean:
        frame[column] = frame[column].apply(normalize_value)

    # ---------------------------------------------------------
    # 5. Reject invalid rows before Neo4j
    # ---------------------------------------------------------

    invalid_rows = frame[
        (frame["chapter_number"] == "")
        | (frame["chapter_name"] == "")
        | (frame["section_number"] == "")
        | (frame["section_name"] == "")
    ]

    if not invalid_rows.empty:
        print("\nInvalid rows detected:")
        print(
            invalid_rows[
                [
                    "chapter_number",
                    "chapter_name",
                    "section_number",
                    "section_name",
                ]
            ].to_string(index=False)
        )

        raise ValueError(
            f"{len(invalid_rows)} invalid rows detected. "
            "Neo4j import stopped."
        )

    # ---------------------------------------------------------
    # 6. Prepare clean records
    # ---------------------------------------------------------

    rows = frame[
        [
            "chapter_number",
            "chapter_name",
            "chapter_subtype",
            "section_number",
            "section_name",
            "description",
        ]
    ].to_dict("records")

    print(f"\nValidated {len(rows)} rows for Neo4j import")

    # ---------------------------------------------------------
    # 7. Connect
    # ---------------------------------------------------------

    driver = GraphDatabase.driver(
        args.uri,
        auth=(args.user, args.password),
    )

    driver.verify_connectivity()

    print("Neo4j connection successful")

    # ---------------------------------------------------------
    # 8. Constraints
    # ---------------------------------------------------------

    constraints = [
        """
        CREATE CONSTRAINT chapter_number_unique IF NOT EXISTS
        FOR (c:Chapter)
        REQUIRE c.chapter_number IS UNIQUE
        """,
        """
        CREATE CONSTRAINT section_number_unique IF NOT EXISTS
        FOR (s:Section)
        REQUIRE s.section_number IS UNIQUE
        """,
    ]

    # ---------------------------------------------------------
    # 9. Import graph
    # ---------------------------------------------------------

    import_query = """
    UNWIND $rows AS row

    MERGE (c:Chapter {
        chapter_number: row.chapter_number
    })

    SET
        c.name = row.chapter_name

    MERGE (s:Section {
        section_number: row.section_number
    })

    SET
        s.name = row.section_name,
        s.description = row.description,
        s.chapter_subtype = row.chapter_subtype

    MERGE (s)-[:BELONGS_TO]->(c)
    """

    try:

        with driver.session() as session:

            for constraint in constraints:
                session.run(constraint).consume()

            session.run(
                import_query,
                rows=rows,
            ).consume()

        print("\nNeo4j import completed successfully.")

    finally:

        driver.close()


if __name__ == "__main__":
    main()
