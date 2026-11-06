import { useMutation } from "@tanstack/react-query";
import { Camera, CircleStop, FileText, Loader2, MonitorPlay, PlusCircle, UserRound } from "lucide-react";
import { useMemo, useState } from "react";
import { Link } from "react-router";

import { api, streamUrl } from "../api/client";
import type { LiveStatus, SessionInfo } from "../api/types";
import { EventList, type EventItem } from "../components/EventList";
import { Badge, Button, Card, CardHeader, ConfirmDialog, EmptyState, ErrorBanner, PageHeader, StatusBadge, cx } from "../components/ui";
import { useLive } from "../live/LiveProvider";
import { examRules, formatDuration, isIncident, secondsBetween, shortId, sourceLabel } from "../lib/format";
import { useNow } from "../lib/useNow";

export function LivePage() {
  const { status, connection } = useLive();

  if (status === null) {
    return (
      <div className="flex items-center gap-2 text-sm text-slate-500">
        <Loader2 className="size-4 animate-spin" aria-hidden />
        {connection === "reconnecting" ? "Waiting for the server…" : "Connecting…"}
      </div>
    );
  }
  if (status.state === "idle" || status.session === null) return <Idle last={status.last_session} />;
  return <ActiveSession session={status.session} live={status.live} />;
}

// --------------------------------------------------------------------- idle
function Idle({ last }: { last: SessionInfo | null }) {
  return (
    <>
      <PageHeader title="Live monitor" />
      <Card>
        <EmptyState icon={<MonitorPlay className="size-10" />} title="No exam session is running">
          <p>Start a session to watch the camera feed, see who is present and get alerts as they happen.</p>
          <Link
            to="/sessions/new"
            className="mt-4 inline-flex items-center gap-2 rounded-lg bg-brand-700 px-3.5 py-2 text-sm font-medium text-white hover:bg-brand-800"
          >
            <PlusCircle className="size-4" aria-hidden /> Start a session
          </Link>
        </EmptyState>
      </Card>
      {last && (
        <Card className="mt-4">
          <CardHeader title="Last session" action={<StatusBadge status={last.state} />} />
          <div className="space-y-2 px-4 py-3 text-sm">
            <p>
              <span className="font-mono text-xs text-slate-500">{shortId(last.session_id)}</span> ·{" "}
              {last.student_names.join(", ")}
            </p>
            {last.error && <ErrorBanner title="The session stopped with an error" error={last.error} />}
            <Link to={`/sessions/${last.session_id}`} className="inline-flex items-center gap-1.5 font-medium text-brand-700 hover:underline dark:text-brand-500">
              <FileText className="size-4" aria-hidden /> View report
            </Link>
          </div>
        </Card>
      )}
    </>
  );
}

// ------------------------------------------------------------------- active
function ActiveSession({ session, live }: { session: SessionInfo; live: LiveStatus | null }) {
  const now = useNow();
  const [confirming, setConfirming] = useState(false);
  const stop = useMutation({ mutationFn: () => api.stopSession(session.session_id), onSettled: () => setConfirming(false) });
  const running = session.state === "running";

  return (
    <>
      <PageHeader
        title={
          <span className="flex items-center gap-3">
            Live monitor <StatusBadge status={session.state} />
          </span>
        }
        description={
          <span className="flex flex-wrap items-center gap-x-3 gap-y-1">
            <span className="font-mono text-xs">{shortId(session.session_id)}</span>
            <span>{sourceLabel(session.source)}</span>
            <span>{examRules(session.exam_context).join(" · ")}</span>
            <span className="font-mono">{formatDuration(secondsBetween(session.started_at, null, now))}</span>
          </span>
        }
        actions={
          <Button
            variant="danger"
            icon={<CircleStop className="size-4" />}
            onClick={() => setConfirming(true)}
            disabled={session.state === "stopping"}
          >
            {session.state === "stopping" ? "Stopping…" : "Stop session"}
          </Button>
        }
      />
      {stop.error && <div className="mb-4"><ErrorBanner error={stop.error} /></div>}
      {session.skipped_students.length > 0 && (
        <p className="mb-4 text-sm text-amber-700 dark:text-amber-300">
          Not proctored: {session.skipped_students.map((s) => `${s.id} (${s.reason})`).join("; ")}
        </p>
      )}

      <div className="grid gap-4 xl:grid-cols-3">
        <div className="space-y-4 xl:col-span-2">
          <VideoCard session={session} live={live} running={running} />
          <PresenceCard names={session.student_names} live={live} />
        </div>
        <div className="space-y-4">
          <AlertFeed sessionId={session.session_id} />
          <PeopleCard live={live} />
        </div>
      </div>

      <ConfirmDialog
        open={confirming}
        title="Stop this session?"
        message="Proctoring stops for every student in this session. The report and evidence are kept."
        confirmLabel="Stop session"
        busy={stop.isPending}
        onConfirm={() => stop.mutate()}
        onCancel={() => setConfirming(false)}
      />
    </>
  );
}

function VideoCard({ session, live, running }: { session: SessionInfo; live: LiveStatus | null; running: boolean }) {
  const [attempt, setAttempt] = useState(0);
  const normal = live?.classification === "Normal";

  return (
    <Card className="overflow-hidden">
      <div className="relative aspect-video bg-slate-950">
        {running ? (
          <img
            key={`${session.session_id}-${attempt}`}
            src={`${streamUrl(session.session_id)}?a=${attempt}`}
            alt="Live annotated camera feed"
            className="size-full object-contain"
            onError={() => setTimeout(() => setAttempt((a) => a + 1), 1000)}
          />
        ) : (
          <div className="flex size-full flex-col items-center justify-center gap-3 text-slate-300">
            <Loader2 className="size-8 animate-spin" aria-hidden />
            <p className="text-sm">
              {session.state === "starting" ? "Loading the detection models and opening the camera…" : "Stopping…"}
            </p>
            {session.state === "starting" && <p className="text-xs text-slate-500">This takes 10–20 seconds on a CPU.</p>}
          </div>
        )}
        {running && live && (
          // Bottom edge: the engine draws its own status text in the frame's top-left corner
          <div className="absolute inset-x-0 bottom-0 flex items-center justify-between gap-2 bg-gradient-to-t from-slate-950/70 to-transparent p-3">
            <Badge tone={normal ? "success" : "danger"} className="text-sm">
              {live.classification}
            </Badge>
            <span className="rounded bg-slate-950/60 px-2 py-0.5 font-mono text-xs text-slate-200">{live.fps.toFixed(1)} fps</span>
          </div>
        )}
      </div>
    </Card>
  );
}

function PresenceCard({ names, live }: { names: string[]; live: LiveStatus | null }) {
  const byName = new Map((live?.students ?? []).map((s) => [s.name, s]));
  return (
    <Card>
      <CardHeader title="Students" subtitle={`${names.length} in this session`} />
      <ul className="grid gap-2 p-4 sm:grid-cols-2 lg:grid-cols-3">
        {names.map((name) => {
          const s = byName.get(name);
          const state = !s?.ever_present ? "Not seen yet" : s.present ? "At desk" : "Away";
          const tone = !s?.ever_present ? "bg-slate-300 dark:bg-slate-600" : s.present ? "bg-emerald-500" : "bg-amber-500";
          return (
            <li key={name} className="flex items-center gap-2.5 rounded-lg border border-slate-200 px-3 py-2 dark:border-slate-800">
              <span className={cx("size-2.5 shrink-0 rounded-full", tone)} aria-hidden />
              <span className="min-w-0 flex-1 truncate text-sm font-medium">{name}</span>
              <span className="text-xs text-slate-500 dark:text-slate-400">{state}</span>
            </li>
          );
        })}
      </ul>
    </Card>
  );
}

function AlertFeed({ sessionId }: { sessionId: string }) {
  const { events } = useLive();
  const [incidentsOnly, setIncidentsOnly] = useState(false);
  const items: EventItem[] = useMemo(
    () =>
      events
        .filter((e) => e.session_id === sessionId && e.level !== "INFO")
        .filter((e) => !incidentsOnly || isIncident(e.level, e.alert_type))
        .map((e, i) => ({ ...e, key: `${e.timestamp}-${i}` }))
        .reverse(),
    [events, sessionId, incidentsOnly],
  );

  return (
    <Card>
      <CardHeader
        title="Alerts"
        subtitle="Newest first"
        action={
          <label className="flex items-center gap-1.5 text-xs text-slate-600 dark:text-slate-300">
            <input type="checkbox" checked={incidentsOnly} onChange={(e) => setIncidentsOnly(e.target.checked)} />
            Incidents only
          </label>
        }
      />
      <div className="max-h-[28rem] overflow-y-auto">
        <EventList items={items} emptyText="No alerts so far." />
      </div>
    </Card>
  );
}

function PeopleCard({ live }: { live: LiveStatus | null }) {
  const people = live?.people ?? [];
  return (
    <Card>
      <CardHeader title="People in view" subtitle={live ? `${people.length} tracked` : undefined} />
      {people.length === 0 ? (
        <p className="px-4 py-6 text-center text-sm text-slate-500 dark:text-slate-400">
          <Camera className="mx-auto mb-1 size-5" aria-hidden /> Nobody in view.
        </p>
      ) : (
        <ul className="divide-y divide-slate-100 dark:divide-slate-800">
          {people.map((p) => (
            <li key={p.track_id} className="px-4 py-2.5 text-sm">
              <div className="flex items-center gap-2">
                <UserRound className="size-4 text-slate-400" aria-hidden />
                <span className="font-medium">{p.name}</span>
                {p.is_photo ? (
                  <Badge tone="danger">Photo</Badge>
                ) : p.liveness === "CONFIRMED_LIVE" ? (
                  <Badge tone="success">Live</Badge>
                ) : (
                  <Badge>Checking liveness</Badge>
                )}
              </div>
              <div className="mt-1 flex flex-wrap gap-x-3 font-mono text-xs text-slate-500 dark:text-slate-400">
                {p.yaw !== null && <span>yaw {p.yaw.toFixed(0)}°</span>}
                {p.pitch !== null && <span>pitch {p.pitch.toFixed(0)}°</span>}
                {p.gaze && <span>gaze {p.gaze}</span>}
              </div>
              {p.alerts.length > 0 && (
                <div className="mt-1.5 flex flex-wrap gap-1">
                  {p.alerts.map((a) => (
                    <Badge key={a} tone="warning">
                      {a}
                    </Badge>
                  ))}
                </div>
              )}
            </li>
          ))}
        </ul>
      )}
    </Card>
  );
}
