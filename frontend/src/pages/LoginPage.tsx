import { ScanEye } from "lucide-react";
import { useState, type FormEvent } from "react";
import { Navigate, useLocation, useNavigate } from "react-router";

import { useLogin, useMe } from "../auth/auth";
import { Button, ErrorBanner } from "../components/ui";

const INPUT =
  "mt-1 block w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm shadow-sm focus:border-brand-600 focus:outline-none focus:ring-2 focus:ring-brand-600/20 dark:border-slate-700 dark:bg-slate-950";

export function LoginPage() {
  const me = useMe();
  const login = useLogin();
  const navigate = useNavigate();
  const location = useLocation();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const from = (location.state as { from?: string } | null)?.from ?? "/live";

  if (me.data) return <Navigate to={from} replace />;

  const submit = (e: FormEvent) => {
    e.preventDefault();
    login.mutate({ username: username.trim(), password }, { onSuccess: () => navigate(from, { replace: true }) });
  };

  return (
    <div className="grid min-h-screen place-items-center px-4">
      <div className="w-full max-w-sm">
        <div className="mb-6 flex flex-col items-center gap-3 text-center">
          <div className="grid size-12 place-items-center rounded-xl bg-brand-700 text-white">
            <ScanEye className="size-7" aria-hidden />
          </div>
          <div>
            <h1 className="text-xl font-semibold tracking-tight">SmartSquad proctor console</h1>
            <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">Sign in to monitor exams and review sessions.</p>
          </div>
        </div>
        <form
          onSubmit={submit}
          className="space-y-4 rounded-xl border border-slate-200 bg-white p-6 shadow-sm dark:border-slate-800 dark:bg-slate-900"
        >
          {login.error && <ErrorBanner error={login.error} />}
          <label className="block text-sm font-medium">
            Username
            <input
              className={INPUT}
              autoComplete="username"
              autoFocus
              required
              value={username}
              onChange={(e) => setUsername(e.target.value)}
            />
          </label>
          <label className="block text-sm font-medium">
            Password
            <input
              className={INPUT}
              type="password"
              autoComplete="current-password"
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
          </label>
          <Button type="submit" className="w-full" busy={login.isPending}>
            Sign in
          </Button>
        </form>
        <p className="mt-4 text-center text-xs text-slate-500 dark:text-slate-400">
          Accounts are created by an administrator with <code className="font-mono">python -m api.manage create-user</code>.
        </p>
      </div>
    </div>
  );
}
