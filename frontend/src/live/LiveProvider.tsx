import { useQueryClient } from "@tanstack/react-query";
import { createContext, useContext, useEffect, useReducer, type ReactNode } from "react";

import type { ServerMessage } from "../api/types";
import { ME_KEY } from "../auth/auth";
import { initialLiveState, liveReducer, type LiveState } from "./liveState";

const LiveContext = createContext<LiveState | null>(null);

const CLOSE_UNAUTHORIZED = 4401;
const MAX_BACKOFF_MS = 10_000;

function socketUrl(): string {
  const scheme = window.location.protocol === "https:" ? "wss" : "ws";
  return `${scheme}://${window.location.host}/api/live`;
}

/** One WebSocket per tab: session state, live status and alerts for every page. Reconnects with backoff. */
export function LiveProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(liveReducer, initialLiveState);
  const qc = useQueryClient();

  useEffect(() => {
    let socket: WebSocket | null = null;
    let retryTimer: ReturnType<typeof setTimeout> | undefined;
    let attempt = 0;
    let disposed = false;

    const connect = () => {
      dispatch({ type: "connection", connection: attempt === 0 ? "connecting" : "reconnecting" });
      socket = new WebSocket(socketUrl());

      socket.onmessage = (ev) => {
        let message: ServerMessage;
        try {
          message = JSON.parse(ev.data as string) as ServerMessage;
        } catch {
          return;
        }
        attempt = 0;
        dispatch({ type: "message", message });
        if (message.type === "state") {
          // Session lists and reports change when a session starts or finishes.
          void qc.invalidateQueries({ queryKey: ["sessions"] });
          void qc.invalidateQueries({ queryKey: ["session", message.session.session_id] });
        }
      };

      socket.onclose = (ev) => {
        if (disposed) return;
        if (ev.code === CLOSE_UNAUTHORIZED) {
          dispatch({ type: "connection", connection: "unauthorized" });
          qc.setQueryData(ME_KEY, null); // the login expired: back to the login page
          return;
        }
        dispatch({ type: "connection", connection: "reconnecting" });
        const delay = Math.min(MAX_BACKOFF_MS, 500 * 2 ** attempt);
        attempt += 1;
        retryTimer = setTimeout(connect, delay);
      };
    };

    connect();
    return () => {
      disposed = true;
      clearTimeout(retryTimer);
      socket?.close();
    };
  }, [qc]);

  return <LiveContext.Provider value={state}>{children}</LiveContext.Provider>;
}

export function useLive(): LiveState {
  const ctx = useContext(LiveContext);
  if (!ctx) throw new Error("useLive must be used inside <LiveProvider>");
  return ctx;
}
