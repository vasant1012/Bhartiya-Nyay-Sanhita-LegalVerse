from __future__ import annotations


class Neo4jStore:
    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        database: str = "neo4j",
        enabled: bool = True,
    ):
        self.uri, self.username, self.password = uri, username, password
        self.database, self.enabled = database, enabled
        self.driver = None
        self.error: str | None = None

    @property
    def available(self) -> bool:
        return self.enabled and self.driver is not None

    def connect(self) -> None:
        if not self.enabled or not self.password:
            self.error = "Neo4j disabled or password missing"
            return
        try:
            from neo4j import GraphDatabase

            self.driver = GraphDatabase.driver(
                self.uri, auth=(self.username, self.password)
            )
            self.driver.verify_connectivity()
        except Exception as exc:  # noqa: BLE001
            self.error = str(exc)
            self.driver = None

    def close(self) -> None:
        if self.driver:
            self.driver.close()

    def section_insights(
        self,
        section_number: str,
        limit: int = 5,
    ) -> dict:

        empty_result = {
            "concepts": [],
            "cross_references": [],
            "related_sections": [],
        }

        if not self.available:
            return empty_result

        available_relationships = self.relationship_types()

        result = dict(empty_result)

        # ---------------------------------------
        # Concepts
        # ---------------------------------------

        if "HAS_CONCEPT" in available_relationships:
            query = """
            MATCH
                (s:Section {
                    section_number: $section_number
                })
                -[:HAS_CONCEPT]->
                (c:Concept)

            RETURN DISTINCT
                c.name AS concept

            ORDER BY concept
            """

            with self.driver.session(database=self.database) as session: # type: ignore
                rows = session.run(
                    query,
                    section_number=str(section_number),
                )

                result["concepts"] = [row["concept"] for row in rows if row["concept"]]

        # ---------------------------------------
        # Cross references
        # ---------------------------------------

        if "REFERENCES" in available_relationships:
            query = """
            MATCH
                (s:Section {
                    section_number: $section_number
                })
                -[:REFERENCES]->
                (r:Section)

            RETURN DISTINCT
                r.section_number
                    AS section_number

            ORDER BY section_number
            """

            with self.driver.session(database=self.database) as session: # type: ignore
                rows = session.run(
                    query,
                    section_number=str(section_number),
                )

                result["cross_references"] = [
                    str(row["section_number"]) for row in rows if row["section_number"]
                ]

        # ---------------------------------------
        # Semantic relationships
        # ---------------------------------------

        if "SEMANTICALLY_RELATED" in available_relationships:
            query = """
            MATCH
                (s:Section {
                    section_number: $section_number
                })
                -[rel:SEMANTICALLY_RELATED]->
                (related:Section)

            RETURN
                related.section_number
                    AS section_number,

                rel.score
                    AS score

            ORDER BY
                rel.score DESC

            LIMIT $limit
            """

            with self.driver.session(database=self.database) as session: # type: ignore
                rows = session.run(
                    query,
                    section_number=str(section_number),
                    limit=limit,
                )

                result["related_sections"] = [
                    {
                        "section_number": str(row["section_number"]),
                        "score": row["score"],
                    }
                    for row in rows
                    if row["section_number"]
                ]

        return result

    def relationship_types(self) -> set[str]:
        if not self.available:
            return set()

        query = """
        CALL db.relationshipTypes()
        YIELD relationshipType
        RETURN relationshipType
        """

        with self.driver.session(database=self.database) as session: # type: ignore
            result = session.run(query)

            return {record["relationshipType"] for record in result}
