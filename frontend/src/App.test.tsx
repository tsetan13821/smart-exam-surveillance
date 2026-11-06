import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { act } from "react";
import { MemoryRouter } from "react-router";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { App, createQueryClient } from "./App";
import { errorMessage } from "./api/client";
import type { ActiveStatus, ServerMessage, SessionInfo, Student } from "./api/types";

// ------------------------------------------------------------- fake server
class FakeSocket {
  static instances: FakeSocket[] = [];
  onmessage: ((ev: { data: string }) => void) | null = null;
  onclose: ((ev: { code: number }) => void) | null = null;
  closed = false;
  constructor(public url: string) {
    FakeSocket.instances.push(this);
  }
  close() {
    this.closed = true;
  }
  send(message: ServerMessage) {
    act(() => this.onmessage?.({ data: JSON.stringify(message) }));
  }
  static latest(): FakeSocket {
    return FakeSocket.instances[FakeSocket.instances.length - 1];
  }
}

const idle: ActiveStatus = { state: "idle", session: null, last_session: null, live: null };

const asha: Student = {
  student_id: "1",
  name: "Asha",
  status: "approved",
  encoding_count: 3,
  extracted_id_text: null,
  created_at: "2026-11-01T09:00:00",
  reviewed_at: null,
  reviewed_by: null,
  photos: [],
};

function running(): SessionInfo {
  return {
    session_id: "abcd1234-0000",
    state: "running",
    student_ids: ["1"],
    student_names: ["Asha"],
    skipped_students: [],
    exam_context: { is_open_book: false, is_paper_exam: false, allowed_devices: [] },
    source: "0",
    started_at: new Date().toISOString(),
    ended_at: null,
    error: null,
  };
}

type Handler = (body: unknown) => [number, unknown];

function fakeServer(routes: Record<string, Handler>) {
  const calls: { method: string; path: string; body: unknown }[] = [];
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const path = String(input);
    const method = init?.method ?? "GET";
    const body = init?.body ? JSON.parse(String(init.body)) : undefined;
    calls.push({ method, path, body });
    const handler = routes[`${method} ${path}`] ?? routes[`${method} ${path.split("?")[0]}`];
    const [status, payload] = handler ? handler(body) : [404, { detail: "Not Found" }];
    return new Response(status === 204 ? null : JSON.stringify(payload), {
      status,
      headers: { "Content-Type": "application/json" },
    });
  });
  vi.stubGlobal("fetch", fetchMock);
  return calls;
}

function renderApp(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <App client={createQueryClient()} />
    </MemoryRouter>,
  );
}

beforeEach(() => {
  FakeSocket.instances = [];
  vi.stubGlobal("WebSocket", FakeSocket);
  HTMLDialogElement.prototype.showModal ??= function (this: HTMLDialogElement) {
    this.setAttribute("open", "");
  };
  HTMLDialogElement.prototype.close ??= function (this: HTMLDialogElement) {
    this.removeAttribute("open");
  };
});

// ------------------------------------------------------------------ tests
describe("login", () => {
  it("sends logged-out visitors to the login page and back after signing in", async () => {
    let loggedIn = false;
    const calls = fakeServer({
      "GET /api/auth/me": () => (loggedIn ? [200, { username: "proctor" }] : [401, { detail: "Not logged in." }]),
      "POST /api/auth/login": (body) => {
        const { password } = body as { password: string };
        if (password !== "right-password") return [401, { detail: "Invalid username or password." }];
        loggedIn = true;
        return [200, { username: "proctor" }];
      },
      "GET /api/students": () => [200, []],
    });
    const user = userEvent.setup();
    renderApp("/sessions");

    expect(await screen.findByRole("heading", { name: /proctor console/i })).toBeInTheDocument();
    await user.type(screen.getByLabelText(/username/i), "proctor");
    await user.type(screen.getByLabelText(/password/i), "wrong");
    await user.click(screen.getByRole("button", { name: /sign in/i }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Invalid username or password.");

    await user.clear(screen.getByLabelText(/password/i));
    await user.type(screen.getByLabelText(/password/i), "right-password");
    await user.click(screen.getByRole("button", { name: /sign in/i }));

    // Lands on the page originally asked for
    expect(await screen.findByRole("heading", { name: "Sessions" })).toBeInTheDocument();
    expect(calls.filter((c) => c.path === "/api/auth/login")).toHaveLength(2);
  });

  it("returns to the login page when the live socket reports an expired login", async () => {
    fakeServer({ "GET /api/auth/me": () => [200, { username: "proctor" }], "GET /api/students": () => [200, []] });
    renderApp("/live");
    await waitFor(() => expect(FakeSocket.instances.length).toBeGreaterThan(0));
    act(() => FakeSocket.latest().onclose?.({ code: 4401 }));
    expect(await screen.findByRole("button", { name: /sign in/i })).toBeInTheDocument();
  });
});

describe("live monitor", () => {
  it("shows the running session, its students and alerts as they arrive", async () => {
    fakeServer({ "GET /api/auth/me": () => [200, { username: "proctor" }], "GET /api/students": () => [200, []] });
    renderApp("/live");
    await waitFor(() => expect(FakeSocket.instances.length).toBeGreaterThan(0));

    FakeSocket.latest().send({ type: "snapshot", status: idle, recent: [] });
    expect(await screen.findByText(/no exam session is running/i)).toBeInTheDocument();

    const session = running();
    FakeSocket.latest().send({ type: "state", session });
    FakeSocket.latest().send({
      type: "status",
      state: "running",
      session,
      last_session: null,
      live: {
        classification: "Normal",
        fps: 29.7,
        frames_processed: 100,
        students: [{ name: "Asha", present: true, ever_present: true }],
        people: [],
      },
    });
    expect(await screen.findByAltText(/live annotated camera feed/i)).toHaveAttribute(
      "src",
      expect.stringContaining("/api/sessions/abcd1234-0000/stream"),
    );
    expect(screen.getByText("29.7 fps")).toBeInTheDocument();
    expect(screen.getByText("At desk")).toBeInTheDocument();

    FakeSocket.latest().send({
      type: "log",
      session_id: session.session_id,
      timestamp: new Date().toISOString(),
      level: "EVIDENCE",
      alert_type: "CONTRABAND_DETECTED",
      message: "(Asha) cell phone detected",
      subject: "Asha",
      evidence_url: "/api/evidence/CONTRABAND_DETECTED_20261106_101500_000001.jpg",
    });
    const alert = (await screen.findByText("(Asha) cell phone detected")).closest("li")!;
    expect(within(alert).getByText("Prohibited device")).toBeInTheDocument();
    expect(within(alert).getByRole("button", { name: /view evidence/i })).toBeInTheDocument();
  });

  it("starts a session from the new-session form", async () => {
    const calls = fakeServer({
      "GET /api/auth/me": () => [200, { username: "proctor" }],
      "GET /api/students": () => [200, [asha]],
      "GET /api/sources": () => [200, [{ type: "camera", label: "Camera 0", index: 0, name: null }]],
      "GET /api/exam-options": () => [200, { devices: ["book", "cell phone"] }],
      "POST /api/sessions": () => [202, { ...running(), state: "starting" }],
    });
    const user = userEvent.setup();
    renderApp("/sessions/new");
    await waitFor(() => expect(FakeSocket.instances.length).toBeGreaterThan(0));
    FakeSocket.latest().send({ type: "snapshot", status: idle, recent: [] });

    const start = await screen.findByRole("button", { name: /start session/i });
    expect(start).toBeDisabled(); // nobody selected yet
    await user.click(await screen.findByRole("checkbox", { name: /asha/i }));
    await user.click(screen.getByRole("checkbox", { name: "book" }));
    await user.click(screen.getByRole("checkbox", { name: /paper exam/i }));
    await user.click(screen.getByRole("button", { name: /start session \(1\)/i }));

    await waitFor(() => expect(calls.some((c) => c.method === "POST" && c.path === "/api/sessions")).toBe(true));
    expect(calls.find((c) => c.method === "POST")?.body).toEqual({
      student_ids: ["1"],
      exam_context: { is_open_book: false, is_paper_exam: true, allowed_devices: ["book"] },
      source: { type: "camera", index: 0 },
    });
    expect(await screen.findByRole("heading", { name: /live monitor/i })).toBeInTheDocument();
  });
});

describe("errorMessage", () => {
  it("reads FastAPI's plain and validation errors", () => {
    expect(errorMessage({ detail: "Session busy" }, "x")).toBe("Session busy");
    expect(
      errorMessage({ detail: [{ loc: ["body", "student_ids"], msg: "List should have at least 1 item" }] }, "x"),
    ).toBe("student_ids: List should have at least 1 item");
    expect(errorMessage(null, "fallback")).toBe("fallback");
  });
});
