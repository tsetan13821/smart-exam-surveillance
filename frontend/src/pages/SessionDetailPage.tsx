import { useQuery } from "@tanstack/react-query";
import { ArrowLeft, Clock, ShieldAlert, Users } from "lucide-react";
import { useState, type ReactNode } from "react";
import { Link, useParams } from "react-router";

import { api, ApiError } from "../api/client";
import type { SessionDetail, StudentReport } from "../api/types";
import { EventList } from "../components/EventList";
import { Badge, Card, CardHeader, EmptyState, ErrorBanner, PageHeader, Spinner, StatusBadge, cx } from "../components/ui";
import { alertLabel, examRules, formatDateTime, formatDuration, formatTime, shortId, sourceLabel } from "../lib/format";

export function SessionDetailPage() {
  const { sessionId = "" } = useParams();
  const session = useQuery({ queryKey: ["session", sessionId], queryFn: () => api.session(sessionId) });

  if (session.isPending) return <Spinner />;
  if (session.error) {
    const notFound = session.error instanceof ApiError && session.error.status === 404;
    return (
      <>
        <BackLink />
        {notFound ? (
          <EmptyState title="Session not found">It may have been recorded on another server.</EmptyState>
        ) : (
          <ErrorBanner error={session.error} />
        )}
      </>
    );
  }
  return <Report session={session.data} />;
}

function BackLink() {
  return (
    <Link to="/sessions" className="mb-4 inline-flex items-center gap-1 text-sm text-slate-600 hover:underline dark:text-slate-300">
      <ArrowLeft className="size-4" aria-hidden /> All sessions
    </Link>
  );
}

function Report({ session }: { session: SessionDetail }) {
  const { report } = session;
  const absent = report.students.filter((s) => s.attendance === "absent").length;

  return (
    <>
      <BackLink />
      <PageHeader
        title={
          <span className="flex flex-wrap items-center gap-3">
            Session {shortId(session.session_id)} <StatusBadge status={session.status} />
          </span>
        }
        description={
          <span className="flex flex-wrap gap-x-3 gap-y-1">
            <span>{formatDateTime(session.start_time)}</span>
            <span>{sourceLabel(session.source)}</span>
            <span>{examRules(session.exam_context).join(" · ")}</span>
          </span>
        }
        actions={
          session.status === "running" && (
            <Link to="/live" className="rounded-lg bg-brand-700 px-3.5 py-2 text-sm font-medium text-white hover:bg-brand-800">
              Open live monitor
            </Link>
          )
        }
      />

      <div className="mb-4 grid gap-4 sm:grid-cols-3">
        <Stat icon={<Clock className="size-4" />} label="Duration" value={formatDuration(report.duration_seconds)} />
        <Stat
          icon={<Users className="size-4" />}
          label="Attendance"
          value={`${report.students.length - absent} / ${report.students.length}`}
          hint={absent ? `${absent} never seen` : "everyone attended"}
        />
        <Stat
          icon={<ShieldAlert className="size-4" />}
          label="Incidents"
          value={String(session.incident_count)}
          hint={`${session.evidence_count} screenshots`}
        />
      </div>

      <div className="grid gap-4 xl:grid-cols-3">
        <div className="space-y-4 xl:col-span-2">
          <Card>
            <CardHeader title="Students" />
            <ul className="divide-y divide-slate-100 dark:divide-slate-800">
              {report.students.map((s) => (
                <StudentRow key={s.student_id} student={s} />
              ))}
            </ul>
          </Card>
          <Timeline sessionId={session.session_id} />
        </div>

        <div className="space-y-4">
          <Card>
            <CardHeader title="Incidents by type" />
            {Object.keys(report.incidents_by_type).length === 0 ? (
              <p className="px-4 py-6 text-center text-sm text-slate-500">No incidents recorded.</p>
            ) : (
              <ul className="space-y-2 px-4 py-3 text-sm">
                {Object.entries(report.incidents_by_type).map(([type, n]) => (
                  <li key={type} className="flex items-center justify-between gap-2">
                    <span>{alertLabel(type)}</span>
                    <Badge tone="warning">{n}</Badge>
                  </li>
                ))}
              </ul>
            )}
            {Object.keys(report.unassigned_incidents).length > 0 && (
              <p className="border-t border-slate-200 px-4 py-2 text-xs text-slate-500 dark:border-slate-800">
                Not tied to a student:{" "}
                {Object.entries(report.unassigned_incidents)
                  .map(([t, n]) => `${alertLabel(t)} ×${n}`)
                  .join(", ")}
              </p>
            )}
          </Card>
          {report.system_errors > 0 && (
            <ErrorBanner title="System errors" error={`${report.system_errors} error(s) were logged during this session; see the timeline.`} />
          )}
        </div>
      </div>
    </>
  );
}

function Stat({ icon, label, value, hint }: { icon: ReactNode; label: string; value: string; hint?: string }) {
  return (
    <Card className="px-4 py-3">
      <p className="flex items-center gap-1.5 text-xs font-medium uppercase tracking-wide text-slate-500 dark:text-slate-400">
        {icon}
        {label}
      </p>
      <p className="mt-1 text-2xl font-semibold tabular-nums">{value}</p>
      {hint && <p className="text-xs text-slate-500 dark:text-slate-400">{hint}</p>}
    </Card>
  );
}

function StudentRow({ student }: { student: StudentReport }) {
  return (
    <li className="px-4 py-3">
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-medium">{student.name}</span>
        <span className="font-mono text-xs text-slate-500">{student.student_id}</span>
        <StatusBadge status={student.attendance} />
        {student.incident_count > 0 && <Badge tone="warning">{student.incident_count} incidents</Badge>}
      </div>
      <dl className="mt-2 grid gap-x-6 gap-y-1 text-sm sm:grid-cols-3">
        <div>
          <dt className="text-xs text-slate-500 dark:text-slate-400">First seen</dt>
          <dd className="font-mono text-xs">{formatTime(student.first_seen)}</dd>
        </div>
        <div>
          <dt className="text-xs text-slate-500 dark:text-slate-400">Time away</dt>
          <dd className="font-mono text-xs">
            {formatDuration(student.away_seconds)} ({student.breaks.length} break{student.breaks.length === 1 ? "" : "s"})
          </dd>
        </div>
        <div>
          <dt className="text-xs text-slate-500 dark:text-slate-400">Incidents</dt>
          <dd className="text-xs">
            {Object.entries(student.incidents)
              .map(([t, n]) => `${alertLabel(t)} ×${n}`)
              .join(", ") || "None"}
          </dd>
        </div>
      </dl>
      {student.breaks.length > 0 && (
        <ul className="mt-2 flex flex-wrap gap-1.5">
          {student.breaks.map((b) => (
            <li key={b.left} className="rounded bg-slate-100 px-2 py-0.5 font-mono text-xs dark:bg-slate-800">
              {formatTime(b.left)} → {b.returned ? formatTime(b.returned) : "end"} ({formatDuration(b.seconds)})
            </li>
          ))}
        </ul>
      )}
    </li>
  );
}

const TIMELINE_FILTERS = [
  { key: "incidents", label: "Incidents" },
  { key: "evidence", label: "All evidence" },
  { key: "all", label: "Everything" },
] as const;
type TimelineFilter = (typeof TIMELINE_FILTERS)[number]["key"];

function Timeline({ sessionId }: { sessionId: string }) {
  const [filter, setFilter] = useState<TimelineFilter>("incidents");
  const alerts = useQuery({
    queryKey: ["session", sessionId, "alerts", filter],
    queryFn: () =>
      api.alerts(sessionId, {
        limit: 1000,
        incidents_only: filter === "incidents" || undefined,
        level: filter === "evidence" ? ["EVIDENCE", "ALERT"] : undefined,
      }),
  });

  return (
    <Card>
      <CardHeader
        title="Timeline"
        subtitle={alerts.data ? `${alerts.data.total} events` : undefined}
        action={
          <div role="group" aria-label="Timeline filter" className="flex gap-1">
            {TIMELINE_FILTERS.map((f) => (
              <button
                key={f.key}
                type="button"
                aria-pressed={filter === f.key}
                onClick={() => setFilter(f.key)}
                className={cx(
                  "rounded-md px-2 py-1 text-xs",
                  filter === f.key ? "bg-slate-900 text-white dark:bg-slate-100 dark:text-slate-900" : "hover:bg-slate-100 dark:hover:bg-slate-800",
                )}
              >
                {f.label}
              </button>
            ))}
          </div>
        }
      />
      {alerts.isPending ? (
        <div className="p-4">
          <Spinner />
        </div>
      ) : alerts.error ? (
        <div className="p-4">
          <ErrorBanner error={alerts.error} />
        </div>
      ) : (
        <div className="max-h-[36rem] overflow-y-auto">
          <EventList
            items={alerts.data.items.map((a) => ({ ...a, key: String(a.alert_id) }))}
            emptyText={filter === "incidents" ? "No incidents in this session." : "No events."}
          />
        </div>
      )}
    </Card>
  );
}
