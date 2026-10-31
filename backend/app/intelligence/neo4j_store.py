from __future__ import annotations

class Neo4jStore:
    def __init__(self, uri: str, username: str, password: str, database: str = 'neo4j', enabled: bool = True):
        self.uri, self.username, self.password = uri, username, password
        self.database, self.enabled = database, enabled
        self.driver = None
        self.error: str | None = None

    @property
    def available(self) -> bool:
        return self.enabled and self.driver is not None

    def connect(self) -> None:
        if not self.enabled or not self.password:
            self.error = 'Neo4j disabled or password missing'
            return
        try:
            from neo4j import GraphDatabase
            self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            self.driver.verify_connectivity()
        except Exception as exc:
            self.error = str(exc)
            self.driver = None

    def close(self) -> None:
        if self.driver:
            self.driver.close()

    def section_insights(self, section_number: str, limit: int = 5) -> dict:
        if not self.available:
            return {'concepts': [], 'cross_references': [], 'related_sections': []}
        query = '''
        MATCH (s:Section {section_number: $section_number})
        OPTIONAL MATCH (s)-[:HAS_CONCEPT]->(c:Concept)
        OPTIONAL MATCH (s)-[:REFERENCES]->(r:Section)
        OPTIONAL MATCH (s)-[rel:SEMANTICALLY_RELATED]->(x:Section)
        RETURN collect(DISTINCT c.name) AS concepts,
               collect(DISTINCT r.section_number) AS cross_references,
               collect(DISTINCT {section_number:x.section_number, score:rel.score})[0..$limit] AS related
        '''
        with self.driver.session(database=self.database) as session:
            row = session.run(query, section_number=str(section_number), limit=limit).single()
            if not row:
                return {'concepts': [], 'cross_references': [], 'related_sections': []}
            return {
                'concepts': [x for x in row['concepts'] if x],
                'cross_references': [x for x in row['cross_references'] if x],
                'related_sections': [x for x in row['related'] if x.get('section_number')],
            }
