import type {
  ActiveStatus,
  AlertLevel,
  AlertRecord,
  ExamContext,
  Page,
  SessionDetail,
  SessionInfo,
  SessionSummary,
  SourceOption,
  SourceSpec,
  Student,
  StudentStatus,
} from "./types";

export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

type ValidationIssue = { loc?: (string | number)[]; msg?: string };

/** FastAPI errors are {"detail": "..."} or, for invalid input, {"detail": [{loc, msg}, ...]}. */
export function errorMessage(body: unknown, fallback: string): string {
  const detail = (body as { detail?: unknown } | null)?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return (detail as ValidationIssue[])
      .map((d) => {
        const field = d.loc?.filter((p) => p !== "body").join(".");
        return field ? `${field}: ${d.msg}` : (d.msg ?? "invalid value");
      })
      .join("; ");
  }
  return fallback;
}

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  const res = await fetch(path, {
    method,
    credentials: "same-origin",
    headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!res.ok) {
    let payload: unknown = null;
    try {
      payload = await res.json();
    } catch {
      /* not JSON */
    }
    throw new ApiError(res.status, errorMessage(payload, `${res.status} ${res.statusText}`));
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

function query(params: Record<string, string | number | boolean | string[] | undefined | null>): string {
  const q = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null || value === "") continue;
    if (Array.isArray(value)) value.forEach((v) => q.append(key, v));
    else q.set(key, String(value));
  }
  const s = q.toString();
  return s ? `?${s}` : "";
}

export interface StartSessionBody {
  student_ids: string[];
  exam_context: ExamContext;
  source: SourceSpec | null;
}

export const api = {
  me: () => request<{ username: string }>("GET", "/api/auth/me"),
  login: (username: string, password: string) =>
    request<{ username: string }>("POST", "/api/auth/login", { username, password }),
  logout: () => request<void>("POST", "/api/auth/logout"),

  students: (status?: StudentStatus) => request<Student[]>("GET", `/api/students${query({ status })}`),
  setStudentStatus: (id: string, status: StudentStatus) =>
    request<Student>("PATCH", `/api/students/${encodeURIComponent(id)}`, { status }),

  sources: () => request<SourceOption[]>("GET", "/api/sources"),
  examOptions: () => request<{ devices: string[] }>("GET", "/api/exam-options"),

  active: () => request<ActiveStatus>("GET", "/api/sessions/active"),
  startSession: (body: StartSessionBody) => request<SessionInfo>("POST", "/api/sessions", body),
  stopSession: (id: string) => request<SessionInfo>("POST", `/api/sessions/${encodeURIComponent(id)}/stop`),

  sessions: (params: { limit?: number; offset?: number; status?: string; student_id?: string }) =>
    request<Page<SessionSummary>>("GET", `/api/sessions${query(params)}`),
  session: (id: string) => request<SessionDetail>("GET", `/api/sessions/${encodeURIComponent(id)}`),
  alerts: (id: string, params: { limit?: number; offset?: number; level?: AlertLevel[]; incidents_only?: boolean }) =>
    request<Page<AlertRecord>>("GET", `/api/sessions/${encodeURIComponent(id)}/alerts${query(params)}`),
};

export const streamUrl = (sessionId: string) => `/api/sessions/${encodeURIComponent(sessionId)}/stream`;
