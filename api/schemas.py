# api/schemas.py
"""Request bodies and the main response shapes (also documents the API in /docs)."""
from typing import Annotated, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ------------------------------------------------------------------ auth
class LoginRequest(_Strict):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class User(BaseModel):
    username: str


# -------------------------------------------------------------- students
class Student(BaseModel):
    student_id: str
    name: str
    status: Literal["pending", "approved", "rejected"]
    encoding_count: int
    extracted_id_text: Optional[str] = None
    created_at: str
    reviewed_at: Optional[str] = None
    reviewed_by: Optional[str] = None
    photos: List[str] = []              # URLs of the registration photos that exist


class StudentStatusUpdate(_Strict):
    status: Literal["approved", "rejected", "pending"]


# --------------------------------------------------------------- sources
class CameraSource(_Strict):
    type: Literal["camera"]
    index: int = Field(default=0, ge=0, le=16)


class VideoSource(_Strict):
    type: Literal["video"]
    name: str = Field(min_length=1, max_length=200)


SourceSpec = Annotated[Union[CameraSource, VideoSource], Field(discriminator="type")]


class SourceOption(BaseModel):
    type: Literal["camera", "video"]
    label: str
    index: Optional[int] = None
    name: Optional[str] = None


# -------------------------------------------------------------- sessions
class ExamContext(_Strict):
    is_open_book: bool = False
    is_paper_exam: bool = False
    allowed_devices: List[str] = []


class StartSessionRequest(_Strict):
    student_ids: List[str] = Field(min_length=1, max_length=50)
    exam_context: ExamContext = ExamContext()
    source: Optional[SourceSpec] = None     # default: the configured camera


class SkippedStudent(BaseModel):
    id: str
    reason: str


class SessionInfo(BaseModel):
    session_id: str
    state: str
    student_ids: List[str]
    student_names: List[str]
    skipped_students: List[SkippedStudent] = []
    exam_context: dict
    source: Optional[str] = None
    started_at: str
    ended_at: Optional[str] = None
    error: Optional[str] = None


class ActiveStatus(BaseModel):
    state: str
    session: Optional[SessionInfo] = None
    last_session: Optional[SessionInfo] = None
    live: Optional[dict] = None


class SessionSummary(BaseModel):
    session_id: str
    student_ids: List[str]
    student_names: List[str]
    start_time: str
    end_time: Optional[str] = None
    status: str
    exam_context: dict
    source: Optional[str] = None
    incident_count: int
    evidence_count: int


class SessionPage(BaseModel):
    items: List[SessionSummary]
    total: int
    limit: int
    offset: int


class AlertRecord(BaseModel):
    alert_id: int
    timestamp: str
    level: str
    alert_type: Optional[str] = None
    message: str
    subject: Optional[str] = None
    evidence_url: Optional[str] = None


class AlertPage(BaseModel):
    items: List[AlertRecord]
    total: int
    limit: int
    offset: int


class SessionDetail(SessionSummary):
    report: dict
