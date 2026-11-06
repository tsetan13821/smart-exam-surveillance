import { useMutation, useQuery } from "@tanstack/react-query";
import { Camera, Film, Play, Search, Users } from "lucide-react";
import { useMemo, useState, type FormEvent } from "react";
import { Link, useNavigate } from "react-router";

import { api, ApiError } from "../api/client";
import type { SourceOption, SourceSpec } from "../api/types";
import { Button, Card, CardHeader, EmptyState, ErrorBanner, PageHeader, Spinner, cx } from "../components/ui";
import { useLive } from "../live/LiveProvider";

function sourceKey(s: SourceOption): string {
  return s.type === "camera" ? `camera:${s.index}` : `video:${s.name}`;
}

function toSpec(s: SourceOption): SourceSpec {
  return s.type === "camera" ? { type: "camera", index: s.index ?? 0 } : { type: "video", name: s.name ?? "" };
}

export function NewSessionPage() {
  const navigate = useNavigate();
  const live = useLive();
  const students = useQuery({ queryKey: ["students", "approved"], queryFn: () => api.students("approved") });
  const sources = useQuery({ queryKey: ["sources"], queryFn: api.sources });
  const options = useQuery({ queryKey: ["exam-options"], queryFn: api.examOptions, staleTime: Infinity });

  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [filter, setFilter] = useState("");
  const [openBook, setOpenBook] = useState(false);
  const [paperExam, setPaperExam] = useState(false);
  const [allowed, setAllowed] = useState<Set<string>>(new Set());
  const [source, setSource] = useState<string | null>(null);

  const start = useMutation({
    mutationFn: api.startSession,
    onSuccess: () => navigate("/live"),
  });

  const visible = useMemo(() => {
    const q = filter.trim().toLowerCase();
    return (students.data ?? []).filter(
      (s) => !q || s.name.toLowerCase().includes(q) || s.student_id.toLowerCase().includes(q),
    );
  }, [students.data, filter]);

  const sourceList = sources.data ?? [];
  const chosenSource = sourceList.find((s) => sourceKey(s) === source) ?? sourceList[0];
  const busy = live.status !== null && live.status.state !== "idle";

  const toggle = (set: Set<string>, value: string) => {
    const next = new Set(set);
    if (next.has(value)) next.delete(value);
    else next.add(value);
    return next;
  };

  const allVisibleSelected = visible.length > 0 && visible.every((s) => selected.has(s.student_id));

  const submit = (e: FormEvent) => {
    e.preventDefault();
    start.mutate({
      student_ids: [...selected],
      exam_context: { is_open_book: openBook, is_paper_exam: paperExam, allowed_devices: [...allowed] },
      source: chosenSource ? toSpec(chosenSource) : null,
    });
  };

  return (
    <form onSubmit={submit}>
      <PageHeader
        title="New exam session"
        description="Choose the students, the exam rules and the camera. Only approved registrations can be proctored."
        actions={
          <Button type="submit" icon={<Play className="size-4" />} busy={start.isPending} disabled={selected.size === 0 || busy}>
            Start session{selected.size > 0 && ` (${selected.size})`}
          </Button>
        }
      />

      {busy && (
        <div className="mb-4 rounded-lg border border-sky-200 bg-sky-50 px-3 py-2 text-sm text-sky-800 dark:border-sky-900 dark:bg-sky-950 dark:text-sky-200">
          A session is already running. There is one camera, so{" "}
          <Link to="/live" className="font-medium underline">
            stop it on the live monitor
          </Link>{" "}
          before starting another.
        </div>
      )}
      {start.error && (
        <div className="mb-4">
          <ErrorBanner
            title={start.error instanceof ApiError && start.error.status === 409 ? "Another session is running" : "Could not start the session"}
            error={start.error}
          />
        </div>
      )}

      <div className="grid gap-4 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader
            title="Students"
            subtitle={`${selected.size} selected`}
            action={
              visible.length > 0 && (
                <button
                  type="button"
                  className="text-sm font-medium text-brand-700 hover:underline dark:text-brand-500"
                  onClick={() =>
                    setSelected((prev) => {
                      const next = new Set(prev);
                      visible.forEach((s) => (allVisibleSelected ? next.delete(s.student_id) : next.add(s.student_id)));
                      return next;
                    })
                  }
                >
                  {allVisibleSelected ? "Clear" : "Select all"}
                </button>
              )
            }
          />
          <div className="border-b border-slate-200 px-4 py-2 dark:border-slate-800">
            <label className="flex items-center gap-2">
              <Search className="size-4 text-slate-400" aria-hidden />
              <span className="sr-only">Filter students</span>
              <input
                value={filter}
                onChange={(e) => setFilter(e.target.value)}
                placeholder="Filter by name or ID"
                className="w-full bg-transparent py-1 text-sm outline-none"
              />
            </label>
          </div>
          {students.isPending ? (
            <div className="p-4">
              <Spinner />
            </div>
          ) : students.error ? (
            <div className="p-4">
              <ErrorBanner error={students.error} />
            </div>
          ) : visible.length === 0 ? (
            <EmptyState icon={<Users className="size-8" />} title={filter ? "No matching students" : "No approved students"}>
              {!filter && (
                <>
                  Approve registrations on the <Link to="/students" className="underline">Students</Link> page first.
                </>
              )}
            </EmptyState>
          ) : (
            <ul className="max-h-[26rem] divide-y divide-slate-100 overflow-y-auto dark:divide-slate-800">
              {visible.map((s) => (
                <li key={s.student_id}>
                  <label className="flex cursor-pointer items-center gap-3 px-4 py-2.5 hover:bg-slate-50 dark:hover:bg-slate-800/50">
                    <input
                      type="checkbox"
                      checked={selected.has(s.student_id)}
                      onChange={() => setSelected((prev) => toggle(prev, s.student_id))}
                    />
                    {s.photos[0] ? (
                      <img src={s.photos[0]} alt="" className="size-8 rounded-full object-cover" />
                    ) : (
                      <span className="grid size-8 place-items-center rounded-full bg-slate-200 text-xs font-semibold dark:bg-slate-700">
                        {s.name.charAt(0).toUpperCase()}
                      </span>
                    )}
                    <span className="flex-1 text-sm font-medium">{s.name}</span>
                    <span className="font-mono text-xs text-slate-500">{s.student_id}</span>
                  </label>
                </li>
              ))}
            </ul>
          )}
        </Card>

        <div className="space-y-4">
          <Card>
            <CardHeader title="Exam rules" />
            <div className="space-y-3 px-4 py-3 text-sm">
              <label className="flex items-start gap-2">
                <input type="checkbox" className="mt-0.5" checked={openBook} onChange={(e) => setOpenBook(e.target.checked)} />
                <span>
                  <span className="font-medium">Open book</span>
                  <span className="block text-xs text-slate-500 dark:text-slate-400">Books are allowed on the desk.</span>
                </span>
              </label>
              <label className="flex items-start gap-2">
                <input type="checkbox" className="mt-0.5" checked={paperExam} onChange={(e) => setPaperExam(e.target.checked)} />
                <span>
                  <span className="font-medium">Paper exam</span>
                  <span className="block text-xs text-slate-500 dark:text-slate-400">Looking down at the desk is expected.</span>
                </span>
              </label>
              <fieldset>
                <legend className="font-medium">Allowed devices</legend>
                <p className="text-xs text-slate-500 dark:text-slate-400">Anything not ticked raises an alert when seen.</p>
                <div className="mt-2 flex flex-wrap gap-2">
                  {(options.data?.devices ?? []).map((d) => (
                    <label
                      key={d}
                      className={cx(
                        "cursor-pointer rounded-full border px-2.5 py-1 text-xs capitalize",
                        allowed.has(d)
                          ? "border-brand-600 bg-brand-50 text-brand-800 dark:bg-brand-800/30 dark:text-brand-100"
                          : "border-slate-300 dark:border-slate-700",
                      )}
                    >
                      <input type="checkbox" className="sr-only" checked={allowed.has(d)} onChange={() => setAllowed((p) => toggle(p, d))} />
                      {d}
                    </label>
                  ))}
                </div>
              </fieldset>
            </div>
          </Card>

          <Card>
            <CardHeader title="Video source" />
            <div className="space-y-2 px-4 py-3">
              {sources.isPending && <Spinner />}
              {sourceList.map((s) => (
                <label
                  key={sourceKey(s)}
                  className={cx(
                    "flex cursor-pointer items-center gap-3 rounded-lg border px-3 py-2 text-sm",
                    chosenSource && sourceKey(chosenSource) === sourceKey(s)
                      ? "border-brand-600 bg-brand-50 dark:bg-brand-800/20"
                      : "border-slate-200 dark:border-slate-800",
                  )}
                >
                  <input
                    type="radio"
                    name="source"
                    checked={chosenSource !== undefined && sourceKey(chosenSource) === sourceKey(s)}
                    onChange={() => setSource(sourceKey(s))}
                  />
                  {s.type === "camera" ? <Camera className="size-4" aria-hidden /> : <Film className="size-4" aria-hidden />}
                  <span className="truncate">{s.label}</span>
                </label>
              ))}
              {sourceList.length === 1 && (
                <p className="text-xs text-slate-500 dark:text-slate-400">
                  To run on a recording instead, put video files in the <code className="font-mono">demo_videos</code> folder.
                </p>
              )}
            </div>
          </Card>
        </div>
      </div>
    </form>
  );
}
