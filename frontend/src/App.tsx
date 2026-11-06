import { MutationCache, QueryCache, QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Link, Navigate, Route, Routes } from "react-router";

import { ApiError } from "./api/client";
import { ME_KEY, RequireAuth } from "./auth/auth";
import { Layout } from "./components/Layout";
import { EmptyState } from "./components/ui";
import { LivePage } from "./pages/LivePage";
import { LoginPage } from "./pages/LoginPage";
import { NewSessionPage } from "./pages/NewSessionPage";
import { SessionDetailPage } from "./pages/SessionDetailPage";
import { SessionsPage } from "./pages/SessionsPage";
import { StudentsPage } from "./pages/StudentsPage";

export function createQueryClient(): QueryClient {
  // Any 401 means the login expired or was revoked: drop the user so RequireAuth shows the login page.
  const onError = (error: unknown) => {
    if (error instanceof ApiError && error.status === 401) client.setQueryData(ME_KEY, null);
  };
  const client: QueryClient = new QueryClient({
    queryCache: new QueryCache({ onError }),
    mutationCache: new MutationCache({ onError }),
    defaultOptions: {
      queries: {
        staleTime: 10_000,
        retry: (count, error) => !(error instanceof ApiError && error.status < 500) && count < 2,
      },
    },
  });
  return client;
}

export function AppRoutes() {
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route element={<RequireAuth />}>
        <Route element={<Layout />}>
          <Route index element={<Navigate to="/live" replace />} />
          <Route path="/live" element={<LivePage />} />
          <Route path="/sessions/new" element={<NewSessionPage />} />
          <Route path="/sessions" element={<SessionsPage />} />
          <Route path="/sessions/:sessionId" element={<SessionDetailPage />} />
          <Route path="/students" element={<StudentsPage />} />
          <Route
            path="*"
            element={
              <EmptyState title="Page not found">
                <Link to="/live" className="underline">
                  Go to the live monitor
                </Link>
              </EmptyState>
            }
          />
        </Route>
      </Route>
    </Routes>
  );
}

export function App({ client }: { client: QueryClient }) {
  return (
    <QueryClientProvider client={client}>
      <AppRoutes />
    </QueryClientProvider>
  );
}
