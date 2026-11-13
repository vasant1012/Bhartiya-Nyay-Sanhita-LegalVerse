def test_get_section(container):
    section=container.sections.get_section('316')
    assert section.section_name == 'Criminal breach of trust'

def test_list_chapters(container):
    chapters=container.sections.list_chapters()
    assert len(chapters)==2
    assert chapters[0].section_count==2
