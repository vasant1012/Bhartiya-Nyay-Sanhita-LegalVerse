import argparse
import sys
from pathlib import Path

import pandas as pd
from neo4j import GraphDatabase

sys.path.append(str(Path.cwd()))
from backend.repositories.csv_repository import ALIASES


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='../data/bns_sections.csv')
    p.add_argument('--uri', default='bolt://localhost:7687')
    p.add_argument('--user', default='neo4j')
    p.add_argument('--password', required=True)
    a = p.parse_args()
    frame = pd.read_csv(a.data, dtype=str).fillna('')
    rename = {}
    for target, candidates in ALIASES.items():
        src = next((c for c in candidates if c in frame.columns), None)
        if src:
            rename[src] = target
    rows = frame.rename(columns=rename).to_dict('records')
    driver = GraphDatabase.driver(a.uri, auth=(a.user, a.password))
    query = '''UNWIND $rows AS row MERGE (c:Chapter {chapter_number:row.chapter_number}) SET c.name=row.chapter_name MERGE (s:Section {section_number:row.section_number}) SET s.name=row.section_name, s.description=row.description, s.chapter_subtype=row.chapter_subtype MERGE (s)-[:BELONGS_TO]->(c)'''
    with driver.session() as session:
        session.run(query, rows=rows).consume()
    driver.close()


if __name__ == '__main__':
    main()
