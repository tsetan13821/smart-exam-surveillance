import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Check, Search, UserRound, Users, X } from "lucide-react";
import { useMemo, useState } from "react";
import { useSearchParams } from "react-router";

import { api } from "../api/client";
import type { Student, StudentStatus } from "../api/types";
import { Button, Card, EmptyState, ErrorBanner, Modal, PageHeader, Spinner, StatusBadge, cx } from "../components/ui";
import { formatDateTime } from "../lib/format";

const TABS: { value: StudentStatus | "all"; label: string }[] = [
  { value: "pending", label: "Pending review" },
  { value: "approved", label: "Approved" },
  { value: "rejected", label: "Rejected" },
  { value: "all", label: "All" },
];

export function StudentsPage() {
  const [params, setParams] = useSearchParams();
  const tab = (params.get("tab") as StudentStatus | "all" | null) ?? "pending";
  const [filter, setFilter] = useState("");
  const students = useQuery({ queryKey: ["students", "all"], queryFn: () => api.students() });

  const counts = useMemo(() => {
    const c: Record<string, number> = { all: 0, pending: 0, approved: 0, rejected: 0 };
    for (const s of students.data ?? []) {
      c.all += 1;
      c[s.status] += 1;
    }
    return c;
  }, [students.data]);

  const visible = useMemo(() => {
    const q = filter.trim().toLowerCase();
    return (students.data ?? [])
      .filter((s) => tab === "all" || s.status === tab)
      .filter((s) => !q || s.name.toLowerCase().includes(q) || s.student_id.toLowerCase().includes(q));
  }, [students.data, tab, filter]);

  return (
    <>
      <PageHeader
        title="Students"
        description="Students register themselves in the registration portal. Check their photos, then approve them so they can be proctored."
      />

      <div className="mb-4 flex flex-wrap items-center gap-3">
        <div role="tablist" aria-label="Registration status" className="flex flex-wrap gap-1">
          {TABS.map((t) => (
            <button
              key={t.value}
              type="button"
              role="tab"
              aria-selected={tab === t.value}
              onClick={() => setParams(t.value === "pending" ? {} : { tab: t.value })}
              className={cx(
                "rounded-lg px-3 py-1.5 text-sm font-medium",
                tab === t.value
                  ? "bg-slate-900 text-white dark:bg-slate-100 dark:text-slate-900"
                  : "text-slate-600 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-800",
              )}
            >
              {t.label} <span className="ml-1 opacity-70">{counts[t.value]}</span>
            </button>
          ))}
        </div>
        <label className="ml-auto flex items-center gap-2 rounded-lg border border-slate-300 bg-white px-3 py-1.5 dark:border-slate-700 dark:bg-slate-900">
          <Search className="size-4 text-slate-400" aria-hidden />
          <span className="sr-only">Filter students</span>
          <input
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            placeholder="Name or ID"
            className="w-40 bg-transparent text-sm outline-none"
          />
        </label>
      </div>

      {students.isPending ? (
        <Spinner />
      ) : students.error ? (
        <ErrorBanner error={students.error} />
      ) : visible.length === 0 ? (
        <Card>
          <EmptyState icon={<Users className="size-8" />} title={tab === "pending" && !filter ? "Nothing to review" : "No students here"}>
            {tab === "pending" && !filter && "New registrations from the portal appear here."}
          </EmptyState>
        </Card>
      ) : (
        <div className="grid gap-4 md:grid-cols-2 2xl:grid-cols-3">
          {visible.map((s) => (
            <StudentCard key={s.student_id} student={s} />
          ))}
        </div>
      )}
    </>
  );
}

function StudentCard({ student }: { student: Student }) {
  const qc = useQueryClient();
  const [photo, setPhoto] = useState<string | null>(null);
  const update = useMutation({
    mutationFn: (status: StudentStatus) => api.setStudentStatus(student.student_id, status),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["students"] }),
  });

  return (
    <Card className="flex flex-col">
      <div className="grid grid-cols-3 gap-1 p-2">
        {["front", "left", "right"].map((pose) => {
          const url = student.photos.find((p) => p.endsWith(`/${pose}`));
          return url ? (
            <button
              key={pose}
              type="button"
              onClick={() => setPhoto(url)}
              className="overflow-hidden rounded-md ring-1 ring-slate-200 hover:ring-brand-600 dark:ring-slate-700"
              aria-label={`${student.name}: ${pose} photo`}
            >
              <img src={url} alt="" loading="lazy" className="aspect-[4/3] w-full object-cover" />
            </button>
          ) : (
            <div
              key={pose}
              className="grid aspect-[4/3] place-items-center rounded-md bg-slate-100 text-slate-400 dark:bg-slate-800"
              title={`No ${pose} photo`}
            >
              <UserRound className="size-5" aria-hidden />
            </div>
          );
        })}
      </div>
      <div className="flex-1 px-4 pb-3">
        <div className="flex items-center gap-2">
          <h3 className="truncate font-semibold">{student.name}</h3>
          <StatusBadge status={student.status} />
        </div>
        <p className="font-mono text-xs text-slate-500">ID {student.student_id}</p>
        <dl className="mt-2 space-y-0.5 text-xs text-slate-500 dark:text-slate-400">
          <div>Registered {formatDateTime(student.created_at)}</div>
          <div>{student.encoding_count} face samples</div>
          {student.reviewed_by && (
            <div>
              {student.status === "approved" ? "Approved" : student.status === "rejected" ? "Rejected" : "Reviewed"} by{" "}
              {student.reviewed_by} · {formatDateTime(student.reviewed_at)}
            </div>
          )}
        </dl>
        {update.error && (
          <div className="mt-2">
            <ErrorBanner error={update.error} />
          </div>
        )}
      </div>
      <div className="flex gap-2 border-t border-slate-200 px-4 py-3 dark:border-slate-800">
        {student.status !== "approved" && (
          <Button
            icon={<Check className="size-4" />}
            busy={update.isPending && update.variables === "approved"}
            disabled={update.isPending}
            onClick={() => update.mutate("approved")}
          >
            Approve
          </Button>
        )}
        {student.status !== "rejected" && (
          <Button
            variant="secondary"
            icon={<X className="size-4" />}
            busy={update.isPending && update.variables === "rejected"}
            disabled={update.isPending}
            onClick={() => update.mutate("rejected")}
          >
            Reject
          </Button>
        )}
      </div>
      <Modal open={photo !== null} onClose={() => setPhoto(null)} title={`${student.name} — registration photo`} wide>
        {photo && <img src={photo} alt={`Registration photo of ${student.name}`} className="w-full rounded-lg" />}
      </Modal>
    </Card>
  );
}
