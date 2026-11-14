from backend.schemas.insight import RelatedSection, SectionInsights
from backend.schemas.section import SectionSummary


class InsightService:
    def __init__(self, repository, neo4j_store):
        self.repository = repository
        self.neo4j_store = neo4j_store

    def for_section(self, section_number: str) -> SectionInsights:
        source = self.repository.get_section(section_number)
        if not source:
            raise KeyError(section_number)
        data = self.neo4j_store.section_insights(section_number)
        warnings, partial = [], False
        if not self.neo4j_store.available:
            partial = True
            warnings.append("Graph insights unavailable")
        related = []
        for item in data.get("related_sections", []):
            target = self.repository.get_section(str(item["section_number"]))
            if target:
                related.append(
                    RelatedSection(
                        section=SectionSummary(**target.model_dump()),
                        score=item.get("score"),
                        relationship="semantic",
                    )
                )
        return SectionInsights(
            section_number=section_number,
            concepts=data.get("concepts", []),
            cross_references=data.get("cross_references", []),
            related_sections=related,
            partial_ml=partial,
            warnings=warnings,
        )
