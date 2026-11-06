// Display helpers shared by the pages.

const LABELS: Record<string, string> = {
  ATTENDANCE_MARKED: "Attendance marked",
  PRESENCE_LEFT: "Left the desk",
  PRESENCE_RETURNED: "Returned",
  STUDENT_MISSING: "Student missing",
  CONTRABAND_DETECTED: "Prohibited device",
  UNAUTHORIZED_FACE_DETECTED: "Unauthorized face",
  UNAUTHORIZED_PERSON_REGISTERED: "Unknown person",
  IMAGE_DETECTED: "Photo spoof",
  SPOOF_DETECTED: "Photo spoof",
  HEAD_TURN_ALERT: "Head turned away",
  EXCESSIVE_MOVEMENT_ALERT: "Excessive movement",
  GAZE_AWAY: "Looking away",
  SUSPICIOUS_DEVICE_GAZE: "Looking at a device",
  CAMERA_ERROR: "Camera error",
  SESSION_FAILED: "Session failed",
};

/** Human label for an alert type: "HEAD_TURN_ALERT" -> "Head turned away". */
export function alertLabel(type: string | null | undefined): string {
  if (!type) return "Event";
  if (LABELS[type]) return LABELS[type];
  const words = type.replace(/_ALERT$/, "").toLowerCase().split("_").filter(Boolean);
  const text = words.join(" ");
  return text.charAt(0).toUpperCase() + text.slice(1);
}

export const INFORMATIONAL = new Set(["ATTENDANCE_MARKED", "PRESENCE_LEFT", "PRESENCE_RETURNED", "ACTIVITY"]);

export type Tone = "neutral" | "info" | "success" | "warning" | "danger";

/** How alarming an event is, for colouring. */
export function eventTone(level: string, type: string | null | undefined): Tone {
  if (level === "ALERT") return "danger";
  if (level === "INFO") return "neutral";
  if (type === "ATTENDANCE_MARKED" || type === "PRESENCE_RETURNED") return "success";
  if (type === "PRESENCE_LEFT") return "info";
  if (type && ["CONTRABAND_DETECTED", "IMAGE_DETECTED", "SPOOF_DETECTED", "UNAUTHORIZED_FACE_DETECTED"].includes(type))
    return "danger";
  return "warning";
}

export function isIncident(level: string, type: string | null | undefined): boolean {
  return level === "EVIDENCE" && !INFORMATIONAL.has(type ?? "");
}

/** 75 -> "1m 15s", 3700 -> "1h 1m". */
export function formatDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return "—";
  const s = Math.max(0, Math.round(seconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  if (h) return `${h}h ${m}m`;
  if (m) return `${m}m ${sec.toString().padStart(2, "0")}s`;
  return `${sec}s`;
}

/** Seconds between two ISO timestamps (to now if the end is missing). */
export function secondsBetween(start: string, end?: string | null, now: number = Date.now()): number {
  const a = new Date(start).getTime();
  const b = end ? new Date(end).getTime() : now;
  return Math.max(0, (b - a) / 1000);
}

export function formatTime(iso: string | null | undefined): string {
  if (!iso) return "—";
  return new Date(iso).toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

export function formatDateTime(iso: string | null | undefined): string {
  if (!iso) return "—";
  return new Date(iso).toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function shortId(id: string): string {
  return id.slice(0, 8);
}

/** "C:\\...\\demo_videos\\clip.mp4" -> "Video: clip.mp4"; "0" -> "Camera 0". */
export function sourceLabel(source: string | null | undefined): string {
  if (source === null || source === undefined || source === "") return "Default camera";
  if (/^\d+$/.test(source)) return `Camera ${source}`;
  const name = source.split(/[\\/]/).pop();
  return `Video: ${name}`;
}

export function examRules(ctx: { is_open_book?: boolean; is_paper_exam?: boolean; allowed_devices?: string[] }): string[] {
  const rules: string[] = [];
  rules.push(ctx.is_open_book ? "Open book" : "Closed book");
  if (ctx.is_paper_exam) rules.push("Paper exam");
  if (ctx.allowed_devices?.length) rules.push(`Allowed: ${ctx.allowed_devices.join(", ")}`);
  return rules;
}
