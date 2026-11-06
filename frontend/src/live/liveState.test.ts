import { describe, expect, it } from "vitest";

import type { ActiveStatus, LogEvent, SessionInfo } from "../api/types";
import { initialLiveState, liveReducer, MAX_EVENTS, type LiveState } from "./liveState";

function session(id: string, state: SessionInfo["state"]): SessionInfo {
  return {
    session_id: id,
    state,
    student_ids: ["1"],
    student_names: ["Asha"],
    skipped_students: [],
    exam_context: { is_open_book: false, is_paper_exam: false, allowed_devices: [] },
    source: "0",
    started_at: "2026-11-06T10:00:00",
    ended_at: null,
    error: null,
  };
}

function log(n: number, sessionId = "s1"): LogEvent {
  return {
    type: "log",
    session_id: sessionId,
    timestamp: `2026-11-06T10:00:${String(n % 60).padStart(2, "0")}.${n}`,
    level: "EVIDENCE",
    alert_type: "HEAD_TURN_ALERT",
    message: `event ${n}`,
  };
}

const idle: ActiveStatus = { state: "idle", session: null, last_session: null, live: null };

function apply(state: LiveState, ...messages: Parameters<typeof liveReducer>[1][]): LiveState {
  return messages.reduce(liveReducer, state);
}

describe("liveReducer", () => {
  it("takes the server's snapshot on connect", () => {
    const s = apply(initialLiveState, { type: "message", message: { type: "snapshot", status: idle, recent: [log(1)] } });
    expect(s.connection).toBe("open");
    expect(s.status?.state).toBe("idle");
    expect(s.events).toHaveLength(1);
  });

  it("follows a session from starting to running to ended", () => {
    let s = apply(initialLiveState, { type: "message", message: { type: "snapshot", status: idle, recent: [log(1, "old")] } });
    s = apply(s, { type: "message", message: { type: "state", session: session("s1", "starting") } });
    expect(s.status?.state).toBe("starting");
    expect(s.events).toEqual([]); // a new session starts with an empty feed

    s = apply(s, { type: "message", message: { type: "state", session: session("s1", "running") } });
    s = apply(s, {
      type: "message",
      message: {
        type: "status",
        state: "running",
        session: session("s1", "running"),
        last_session: null,
        live: { classification: "Normal", fps: 30, frames_processed: 10, students: [], people: [] },
      },
    });
    expect(s.status?.live?.fps).toBe(30);

    s = apply(s, { type: "message", message: log(2) });
    s = apply(s, { type: "message", message: { type: "state", session: session("s1", "ended") } });
    expect(s.status).toEqual({ state: "idle", session: null, last_session: session("s1", "ended"), live: null });
    expect(s.events).toHaveLength(1); // the finished session's alerts stay visible
  });

  it("ignores events it already has after a reconnect", () => {
    let s = apply(initialLiveState, { type: "message", message: { type: "snapshot", status: idle, recent: [log(1), log(2)] } });
    s = apply(s, { type: "message", message: log(2) }, { type: "message", message: log(3) });
    expect(s.events.map((e) => e.message)).toEqual(["event 1", "event 2", "event 3"]);
  });

  it("keeps a bounded number of events", () => {
    let s = initialLiveState;
    for (let i = 0; i < MAX_EVENTS + 25; i++) s = apply(s, { type: "message", message: log(i) });
    expect(s.events).toHaveLength(MAX_EVENTS);
    expect(s.events.at(-1)?.message).toBe(`event ${MAX_EVENTS + 24}`);
  });

  it("tracks the connection", () => {
    expect(apply(initialLiveState, { type: "connection", connection: "reconnecting" }).connection).toBe("reconnecting");
  });
});
