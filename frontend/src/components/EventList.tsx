import { ImageIcon } from "lucide-react";
import { useState } from "react";

import { alertLabel, eventTone, formatTime } from "../lib/format";
import { Badge, Modal } from "./ui";

export interface EventItem {
  key: string;
  timestamp: string;
  level: string;
  alert_type?: string | null;
  message: string;
  subject?: string | null;
  evidence_url?: string | null;
}

/** A list of alerts/events with evidence thumbnails that open full size. */
export function EventList({ items, emptyText = "No events yet." }: { items: EventItem[]; emptyText?: string }) {
  const [open, setOpen] = useState<EventItem | null>(null);

  if (items.length === 0) {
    return <p className="px-4 py-8 text-center text-sm text-slate-500 dark:text-slate-400">{emptyText}</p>;
  }

  return (
    <>
      <ul className="divide-y divide-slate-100 dark:divide-slate-800">
        {items.map((e) => (
          <li key={e.key} className="flex gap-3 px-4 py-3">
            {e.evidence_url ? (
              <button
                type="button"
                onClick={() => setOpen(e)}
                className="shrink-0 overflow-hidden rounded-md ring-1 ring-slate-200 hover:ring-brand-600 dark:ring-slate-700"
                aria-label={`View evidence: ${alertLabel(e.alert_type)} at ${formatTime(e.timestamp)}`}
              >
                <img src={e.evidence_url} alt="" loading="lazy" className="h-14 w-20 object-cover" />
              </button>
            ) : (
              <div className="grid h-14 w-20 shrink-0 place-items-center rounded-md bg-slate-100 text-slate-400 dark:bg-slate-800">
                <ImageIcon className="size-4" aria-hidden />
              </div>
            )}
            <div className="min-w-0 flex-1">
              <div className="flex flex-wrap items-center gap-2">
                <Badge tone={eventTone(e.level, e.alert_type)}>{alertLabel(e.alert_type)}</Badge>
                {e.subject && <span className="text-xs font-medium">{e.subject}</span>}
                <time className="ml-auto font-mono text-xs text-slate-500 dark:text-slate-400" dateTime={e.timestamp}>
                  {formatTime(e.timestamp)}
                </time>
              </div>
              <p className="mt-1 break-words text-sm text-slate-600 dark:text-slate-300">{e.message}</p>
            </div>
          </li>
        ))}
      </ul>
      <Modal open={open !== null} onClose={() => setOpen(null)} title={open ? alertLabel(open.alert_type) : ""} wide>
        {open?.evidence_url && (
          <figure>
            <img src={open.evidence_url} alt={`Evidence: ${open.message}`} className="w-full rounded-lg" />
            <figcaption className="mt-2 text-sm text-slate-600 dark:text-slate-300">
              <span className="font-mono text-xs">{formatTime(open.timestamp)}</span> · {open.message}
            </figcaption>
          </figure>
        )}
      </Modal>
    </>
  );
}
