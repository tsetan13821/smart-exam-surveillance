// Shapes returned by the FastAPI backend (see api/schemas.py).

export type StudentStatus = "pending" | "approved" | "rejected";

export interface Student {
  student_id: string;
  name: string;
  status: StudentStatus;
  encoding_count: number;
  extracted_id_text: string | null;
  created_at: string;
  reviewed_at: string | null;
  reviewed_by: string | null;
  photos: string[];
}

export interface SourceOption {
  type: "camera" | "video";
  label: string;
  index: number | null;
  name: string | null;
}

export type SourceSpec = { type: "camera"; index: number } | { type: "video"; name: string };

export interface ExamContext {
  is_open_book: boolean;
  is_paper_exam: boolean;
  allowed_devices: string[];
}

export type SessionState = "starting" | "running" | "stopping" | "ended" | "failed";

export interface SkippedStudent {
  id: string;
  reason: string;
}

export interface SessionInfo {
  session_id: string;
  state: SessionState;
  student_ids: string[];
  student_names: string[];
  skipped_students: SkippedStudent[];
  exam_context: ExamContext;
  source: string | null;
  started_at: string;
  ended_at: string | null;
  error: string | null;
}

export interface LiveStudent {
  name: string;
  present: boolean;
  ever_present: boolean;
}

export interface LivePerson {
  track_id: number;
  name: string;
  is_photo: boolean;
  liveness: string;
  alerts: string[];
  yaw: number | null;
  pitch: number | null;
  gaze: string | null;
}

export interface LiveStatus {
  classification: string;
  fps: number;
  frames_processed: number;
  heavy_model_updates?: number;
  students: LiveStudent[];
  people: LivePerson[];
}

export interface ActiveStatus {
  state: "idle" | SessionState;
  session: SessionInfo | null;
  last_session: SessionInfo | null;
  live: LiveStatus | null;
}

export type SessionRecordStatus = "running" | "ended" | "failed";

export interface SessionSummary {
  session_id: string;
  student_ids: string[];
  student_names: string[];
  start_time: string;
  end_time: string | null;
  status: SessionRecordStatus;
  exam_context: Partial<ExamContext>;
  source: string | null;
  incident_count: number;
  evidence_count: number;
}

export interface Page<T> {
  items: T[];
  total: number;
  limit: number;
  offset: number;
}

export type AlertLevel = "INFO" | "ALERT" | "EVIDENCE";

export interface AlertRecord {
  alert_id: number;
  timestamp: string;
  level: AlertLevel;
  alert_type: string | null;
  message: string;
  subject: string | null;
  evidence_url: string | null;
}

export interface StudentBreak {
  left: string;
  returned: string | null;
  seconds: number | null;
}

export interface StudentReport {
  student_id: string;
  name: string;
  attendance: "present" | "left" | "absent";
  first_seen: string | null;
  breaks: StudentBreak[];
  away_seconds: number;
  incident_count: number;
  incidents: Record<string, number>;
}

export interface SessionReport {
  duration_seconds: number | null;
  students: StudentReport[];
  incidents_by_type: Record<string, number>;
  unassigned_incidents: Record<string, number>;
  system_errors: number;
  informational_types: string[];
}

export interface SessionDetail extends SessionSummary {
  report: SessionReport;
}

// ---------------------------------------------------------------- /api/live
export interface LogEvent {
  type: "log";
  session_id: string;
  timestamp: string;
  level: AlertLevel;
  alert_type?: string | null;
  message: string;
  subject?: string | null;
  evidence_url?: string | null;
  alert_id?: number;
}

export type ServerMessage =
  | { type: "snapshot"; status: ActiveStatus; recent: LogEvent[] }
  | { type: "state"; session: SessionInfo }
  | LogEvent
  | ({ type: "status" } & ActiveStatus)
  | { type: "ping" };
