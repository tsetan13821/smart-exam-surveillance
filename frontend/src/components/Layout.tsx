import { useQuery } from "@tanstack/react-query";
import { History, LogOut, MonitorPlay, PlusCircle, ScanEye, Users, WifiOff } from "lucide-react";
import type { ReactNode } from "react";
import { NavLink, Outlet, useNavigate } from "react-router";

import { api } from "../api/client";
import { useLogout, useMe } from "../auth/auth";
import { LiveProvider, useLive } from "../live/LiveProvider";
import { cx } from "./ui";

export function Layout() {
  return (
    <LiveProvider>
      <div className="flex min-h-screen flex-col md:flex-row">
        <Sidebar />
        <main className="min-w-0 flex-1 px-4 py-6 md:px-8">
          <ConnectionBanner />
          <Outlet />
        </main>
      </div>
    </LiveProvider>
  );
}

function Sidebar() {
  const me = useMe();
  const logout = useLogout();
  const navigate = useNavigate();
  const live = useLive();
  const pending = useQuery({ queryKey: ["students", "pending"], queryFn: () => api.students("pending") });
  const sessionActive = live.status !== null && live.status.state !== "idle";

  return (
    <aside className="border-b border-slate-200 bg-white md:sticky md:top-0 md:flex md:h-screen md:w-60 md:flex-col md:border-r md:border-b-0 dark:border-slate-800 dark:bg-slate-900">
      <div className="flex items-center gap-2 px-4 py-4">
        <div className="grid size-8 place-items-center rounded-lg bg-brand-700 text-white">
          <ScanEye className="size-5" aria-hidden />
        </div>
        <div>
          <p className="text-sm font-semibold leading-tight">SmartSquad</p>
          <p className="text-xs text-slate-500 dark:text-slate-400">Proctor console</p>
        </div>
        <button
          type="button"
          onClick={() => logout.mutate(undefined, { onSettled: () => navigate("/login") })}
          className="ml-auto rounded-md p-2 text-slate-600 hover:bg-slate-100 md:hidden dark:text-slate-300 dark:hover:bg-slate-800"
          aria-label={`Log out ${me.data?.username ?? ""}`}
        >
          <LogOut className="size-4" aria-hidden />
        </button>
      </div>

      <nav aria-label="Main" className="flex gap-1 overflow-x-auto px-2 pb-2 md:flex-1 md:flex-col md:overflow-visible md:pb-0">
        <NavItem to="/live" icon={<MonitorPlay className="size-4" />}>
          Live monitor
          {sessionActive && <span className="ml-auto size-2 animate-pulse rounded-full bg-emerald-500" aria-label="session running" />}
        </NavItem>
        <NavItem to="/sessions/new" icon={<PlusCircle className="size-4" />}>
          New session
        </NavItem>
        <NavItem to="/sessions" end icon={<History className="size-4" />}>
          Sessions
        </NavItem>
        <NavItem to="/students" icon={<Users className="size-4" />}>
          Students
          {!!pending.data?.length && (
            <span className="ml-auto rounded-full bg-amber-100 px-1.5 text-xs font-semibold text-amber-800 dark:bg-amber-900 dark:text-amber-200">
              {pending.data.length}
            </span>
          )}
        </NavItem>
      </nav>

      <div className="hidden items-center justify-between gap-2 border-t border-slate-200 px-4 py-3 md:flex dark:border-slate-800">
        <span className="truncate text-sm text-slate-600 dark:text-slate-300">{me.data?.username}</span>
        <button
          type="button"
          onClick={() => logout.mutate(undefined, { onSettled: () => navigate("/login") })}
          className="inline-flex items-center gap-1.5 rounded-md px-2 py-1 text-sm text-slate-600 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-800"
        >
          <LogOut className="size-4" aria-hidden /> Log out
        </button>
      </div>
    </aside>
  );
}

function NavItem({ to, icon, end, children }: { to: string; icon: ReactNode; end?: boolean; children: ReactNode }) {
  return (
    <NavLink
      to={to}
      end={end}
      className={({ isActive }) =>
        cx(
          "flex shrink-0 items-center gap-2.5 rounded-lg px-3 py-2 text-sm font-medium transition-colors",
          isActive
            ? "bg-brand-50 text-brand-800 dark:bg-brand-800/30 dark:text-brand-100"
            : "text-slate-600 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-800",
        )
      }
    >
      {icon}
      {children}
    </NavLink>
  );
}

function ConnectionBanner() {
  const { connection } = useLive();
  if (connection !== "reconnecting") return null;
  return (
    <div
      role="status"
      className="mb-4 flex items-center gap-2 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-800 dark:border-amber-900 dark:bg-amber-950 dark:text-amber-200"
    >
      <WifiOff className="size-4" aria-hidden /> Lost the live connection to the server. Reconnecting…
    </div>
  );
}
