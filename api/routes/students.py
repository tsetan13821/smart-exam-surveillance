# api/routes/students.py
from typing import List, Literal, Optional
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse

from app.utils import students_db
from api.deps import current_user
from api.schemas import Student, StudentStatusUpdate

router = APIRouter(prefix="/api/students", tags=["students"], dependencies=[Depends(current_user)])

POSES = ("front", "left", "right")


def _public(rec):
    out = {k: v for k, v in rec.items() if k not in ("encodings", "review_evidence_id")}
    sid = quote(str(rec["student_id"]), safe="")
    out["photos"] = [f"/api/students/{sid}/photos/{pose}" for pose in POSES
                     if students_db.registration_photo_path(rec, pose)]
    return out


def _get_or_404(student_id):
    rec = students_db.get_student(student_id)
    if rec is None:
        raise HTTPException(404, "Student not found.")
    return rec


@router.get("", response_model=List[Student])
def list_students(status: Optional[Literal["pending", "approved", "rejected"]] = Query(None)):
    return [_public(r) for r in students_db.list_students(status)]


@router.get("/{student_id}", response_model=Student)
def get_student(student_id: str):
    return _public(_get_or_404(student_id))


@router.get("/{student_id}/photos/{pose}")
def get_photo(student_id: str, pose: Literal["front", "left", "right"]):
    path = students_db.registration_photo_path(_get_or_404(student_id), pose)
    if path is None:
        raise HTTPException(404, "Photo not found.")
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=300"})


@router.patch("/{student_id}", response_model=Student)
def update_status(student_id: str, body: StudentStatusUpdate, user=Depends(current_user)):
    _get_or_404(student_id)
    students_db.set_status(student_id, body.status, reviewed_by=user["username"])
    return _public(_get_or_404(student_id))
