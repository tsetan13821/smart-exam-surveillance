import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Navigate, Outlet, useLocation } from "react-router";

import { api, ApiError } from "../api/client";
import { Button, ErrorBanner, FullPageSpinner } from "../components/ui";

export const ME_KEY = ["me"] as const;

/** The logged-in proctor, or null. A 401 means "logged out", not an error. */
export function useMe() {
  return useQuery({
    queryKey: ME_KEY,
    queryFn: async () => {
      try {
        return await api.me();
      } catch (e) {
        if (e instanceof ApiError && e.status === 401) return null;
        throw e;
      }
    },
    staleTime: 5 * 60_000,
    retry: false,
  });
}

export function useLogin() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ username, password }: { username: string; password: string }) => api.login(username, password),
    onSuccess: (user) => {
      qc.clear();
      qc.setQueryData(ME_KEY, user);
    },
  });
}

export function useLogout() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: api.logout,
    onSettled: () => {
      qc.clear();
      qc.setQueryData(ME_KEY, null);
    },
  });
}

/** Route guard: renders the page for logged-in proctors, otherwise sends them to /login. */
export function RequireAuth() {
  const me = useMe();
  const location = useLocation();
  if (me.isPending) return <FullPageSpinner label="Checking your session…" />;
  if (me.isError) {
    return (
      <div className="mx-auto mt-24 max-w-md space-y-3 px-4">
        <ErrorBanner title="Cannot reach the SmartSquad server" error={me.error} />
        <Button variant="secondary" onClick={() => void me.refetch()} busy={me.isFetching}>
          Try again
        </Button>
      </div>
    );
  }
  if (!me.data) return <Navigate to="/login" replace state={{ from: location.pathname + location.search }} />;
  return <Outlet />;
}
