import { defineConfig } from "@playwright/test";
import os from "node:os";
import path from "node:path";

// End-to-end tests run the real API (python -m api) serving the built frontend (npm run build first).
//   SMARTSQUAD_PYTHON  Python with the backend requirements installed (default: python)
//   E2E_VIDEO_DIR      folder with a video to proctor; the session test is skipped without it
//   E2E_BROWSER        installed browser channel to use, e.g. msedge or chrome (default: Playwright's Chromium)

const PORT = 8766;
const DATA = path.join(os.tmpdir(), "smartsquad-e2e");
const PYTHON = process.env.SMARTSQUAD_PYTHON ?? "python";
const ROOT = path.resolve(import.meta.dirname, "..");

export const E2E_USER = { username: "e2e-proctor", password: "e2e-password-123" };

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  workers: 1, // one API, one camera: the specs share the server
  timeout: 120_000,
  reporter: [["list"]],
  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
    channel: process.env.E2E_BROWSER || undefined,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  webServer: {
    command: `"${PYTHON}" frontend/e2e/seed_e2e.py && "${PYTHON}" -m api --port ${PORT}`,
    cwd: ROOT,
    url: `http://127.0.0.1:${PORT}/api/health`,
    timeout: 120_000,
    reuseExistingServer: false,
    stdout: "pipe",
    env: {
      ...(process.env as Record<string, string>),
      SMARTSQUAD_DB: path.join(DATA, "e2e.db"),
      SMARTSQUAD_REGISTRATIONS_DIR: path.join(DATA, "registrations"),
      SMARTSQUAD_LOG_DIR: path.join(DATA, "logs"),
      SMARTSQUAD_LEGACY_REGISTRY: path.join(DATA, "no-legacy-registry.json"),
      SMARTSQUAD_DEMO_VIDEOS: process.env.E2E_VIDEO_DIR ?? path.join(DATA, "no-videos"),
      SMARTSQUAD_ADMIN_USER: E2E_USER.username,
      SMARTSQUAD_ADMIN_PASSWORD: E2E_USER.password,
    },
  },
});
