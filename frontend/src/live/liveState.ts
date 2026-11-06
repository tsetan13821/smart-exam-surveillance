import type { ActiveStatus, LogEvent, ServerMessage, SessionInfo } from "../api/types";

export type Connection = "connecting" | "open" | "reconnecting" | "unauthorized";

export interface LiveState {
  connection: Connection;
  status: ActiveStatus | null;
  /** Events of the current (or last) session, oldest first. */
  events: LogEvent[];
}

export const MAX_EVENTS = 300;

export const initialLiveState: LiveState = { connection: "connecting", status: null, events: [] };

export type LiveAction = { type: "connection"; connection: Connection } | { type: "message"; message: ServerMessage };

const FINISHED = new Set(["ended", "failed"]);

function withSession(status: ActiveStatus | null, session: SessionInfo): ActiveStatus {
  if (FINISHED.has(session.state)) {
    return { state: "idle", session: null, last_session: session, live: null };
  }
  const sameSession = status?.session?.session_id === session.session_id;
  return {
    state: session.state,
    session,
    last_session: null,
    // While stopping, keep the last live status so presence doesn't flip to "not seen"
    live: sameSession && (session.state === "running" || session.state === "stopping") ? (status?.live ?? null) : null,
  };
}

function eventKey(e: LogEvent): string {
  return `${e.timestamp}|${e.level}|${e.alert_type ?? ""}|${e.message}`;
}

export function liveReducer(state: LiveState, action: LiveAction): LiveState {
  if (action.type === "connection") return { ...state, connection: action.connection };

  const msg = action.message;
  switch (msg.type) {
    case "snapshot":
      return { connection: "open", status: msg.status, events: msg.recent.slice(-MAX_EVENTS) };

    case "state": {
      const previous = state.status?.session?.session_id ?? state.status?.last_session?.session_id;
      const isNew = msg.session.session_id !== previous;
      return {
        ...state,
        status: withSession(state.status, msg.session),
        events: isNew ? [] : state.events,
      };
    }

    case "status": {
      const { type: _type, ...status } = msg;
      return { ...state, status };
    }

    case "log": {
      // A tab that reconnects gets recent events in its snapshot; skip any we already have.
      const key = eventKey(msg);
      if (state.events.slice(-20).some((e) => eventKey(e) === key)) return state;
      const events = [...state.events, msg];
      return { ...state, events: events.length > MAX_EVENTS ? events.slice(-MAX_EVENTS) : events };
    }

    default:
      return state;
  }
}
