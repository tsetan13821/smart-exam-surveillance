import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { ChevronLeft, ChevronRight, History } from "lucide-react";
import { Link, useNavigate, useSearchParams } from "react-router";

import { api } from "../api/client";
import { Badge, Card, EmptyState, ErrorBanner, PageHeader, Spinner, StatusBadge, cx } from "../components/ui";
import { formatDateTime, formatDuration, secondsBetween, shortId, sourceLabel } from "../lib/format";

const PAGE_SIZE = 20;
const FILTERS = [
  { value: "", label: "All" },
  { value: "running", label: "Running" },
  { value: "ended", label: "Ended" },
  { value: "failed", label: "Failed" },
];

export function SessionsPage() {
  const [params, setParams] = useSearchParams();
  const navigate = useNavigate();
  const status = params.get("status") ?? "";
  const page = Math.max(0, Number(params.get("page") ?? 0) || 0);

  const sessions = useQuery({
    queryKey: ["sessions", { status, page }],
    queryFn: () => api.sessions({ limit: PAGE_SIZE, offset: page * PAGE_SIZE, status: status || undefined }),
    placeholderData: keepPreviousData,
  });

  const setFilter = (next: Record<string, string>) => {
    const p = new URLSearchParams(params);
    for (const [k, v] of Object.entries(next)) {
      if (v) p.set(k, v);
      else p.delete(k);
    }
    setParams(p);
  };

  const total = sessions.data?.total ?? 0;
  const pages = Math.max(1, Math.ceil(total / PAGE_SIZE));

  return (
    <>
      <PageHeader
        title="Sessions"
        description="Every proctored exam, newest first. Open one for its attendance report and evidence."
        actions={
          <div role="group" aria-label="Filter by status" className="flex rounded-lg border border-slate-300 p-0.5 dark:border-slate-700">
            {FILTERS.map((f) => (
              <button
                key={f.value}
                type="button"
                aria-pressed={status === f.value}
                onClick={() => setFilter({ status: f.value, page: "" })}
                className={cx(
                  "rounded-md px-3 py-1 text-sm",
                  status === f.value
                    ? "bg-slate-900 text-white dark:bg-slate-100 dark:text-slate-900"
                    : "text-slate-600 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-800",
                )}
              >
                {f.label}
              </button>
            ))}
          </div>
        }
      />

      <Card className="overflow-hidden">
        {sessions.isPending ? (
          <div className="p-4">
            <Spinner />
          </div>
        ) : sessions.error ? (
          <div className="p-4">
            <ErrorBanner error={sessions.error} />
          </div>
        ) : sessions.data.items.length === 0 ? (
          <EmptyState icon={<History className="size-8" />} title="No sessions yet">
            <Link to="/sessions/new" className="underline">
              Start the first one
            </Link>
            .
          </EmptyState>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="bg-slate-50 text-xs uppercase tracking-wide text-slate-500 dark:bg-slate-800/50 dark:text-slate-400">
                <tr>
                  <th className="px-4 py-2.5 font-medium">Started</th>
                  <th className="px-4 py-2.5 font-medium">Students</th>
                  <th className="px-4 py-2.5 font-medium">Duration</th>
                  <th className="px-4 py-2.5 font-medium">Source</th>
                  <th className="px-4 py-2.5 font-medium">Incidents</th>
                  <th className="px-4 py-2.5 font-medium">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
                {sessions.data.items.map((s) => (
                  <tr
                    key={s.session_id}
                    onClick={() => navigate(`/sessions/${s.session_id}`)}
                    className="cursor-pointer hover:bg-slate-50 dark:hover:bg-slate-800/50"
                  >
                    <td className="whitespace-nowrap px-4 py-3">
                      <Link to={`/sessions/${s.session_id}`} className="font-medium hover:underline" onClick={(e) => e.stopPropagation()}>
                        {formatDateTime(s.start_time)}
                      </Link>
                      <div className="font-mono text-xs text-slate-500">{shortId(s.session_id)}</div>
                    </td>
                    <td className="max-w-xs px-4 py-3">
                      <span className="line-clamp-2">{s.student_names.join(", ")}</span>
                    </td>
                    <td className="whitespace-nowrap px-4 py-3 font-mono text-xs">
                      {s.status === "running" ? "—" : formatDuration(secondsBetween(s.start_time, s.end_time ?? s.start_time))}
                    </td>
                    <td className="whitespace-nowrap px-4 py-3 text-slate-600 dark:text-slate-300">{sourceLabel(s.source)}</td>
                    <td className="px-4 py-3">
                      {s.incident_count > 0 ? <Badge tone="warning">{s.incident_count}</Badge> : <span className="text-slate-400">0</span>}
                    </td>
                    <td className="px-4 py-3">
                      <StatusBadge status={s.status} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {total > PAGE_SIZE && (
          <div className="flex items-center justify-between border-t border-slate-200 px-4 py-2 text-sm dark:border-slate-800">
            <span className="text-slate-500">
              Page {page + 1} of {pages} · {total} sessions
            </span>
            <div className="flex gap-1">
              <button
                type="button"
                aria-label="Previous page"
                disabled={page === 0}
                onClick={() => setFilter({ page: page - 1 > 0 ? String(page - 1) : "" })}
                className="rounded p-1 hover:bg-slate-100 disabled:opacity-40 dark:hover:bg-slate-800"
              >
                <ChevronLeft className="size-4" />
              </button>
              <button
                type="button"
                aria-label="Next page"
                disabled={page + 1 >= pages}
                onClick={() => setFilter({ page: String(page + 1) })}
                className="rounded p-1 hover:bg-slate-100 disabled:opacity-40 dark:hover:bg-slate-800"
              >
                <ChevronRight className="size-4" />
              </button>
            </div>
          </div>
        )}
      </Card>
    </>
  );
}
