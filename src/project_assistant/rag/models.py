from pydantic import BaseModel


class Document(BaseModel):
    doc_id: str
    title: str
    typ: str
    projekt_id: str | None
    abteilung: str
    vertraulichkeit: str
    revision: str
    datum: str
    body: str
    is_current: bool = True  # decided at ingestion, never stored in the file


class Chunk(BaseModel):
    chunk_id: str  # "WPS-014-B#0"
    doc_id: str
    heading: str
    text: str  # what gets embedded: header + content
    metadata: dict[str, str | int | float | bool]
