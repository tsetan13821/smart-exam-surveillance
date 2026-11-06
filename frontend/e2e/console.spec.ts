import { expect, test, type Page } from "@playwright/test";

import { E2E_USER } from "../playwright.config";

async function login(page: Page) {
  await page.goto("/login");
  await page.getByLabel("Username").fill(E2E_USER.username);
  await page.getByLabel("Password").fill(E2E_USER.password);
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(page.getByRole("heading", { name: /live monitor/i })).toBeVisible();
}

test("login is required, wrong passwords are refused, logout ends the session", async ({ page }) => {
  await page.goto("/sessions");
  await expect(page).toHaveURL(/\/login$/);

  await page.getByLabel("Username").fill(E2E_USER.username);
  await page.getByLabel("Password").fill("not-the-password");
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(page.getByRole("alert")).toContainText("Invalid username or password");

  await page.getByLabel("Password").fill(E2E_USER.password);
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(page).toHaveURL(/\/sessions$/); // back to the page first asked for
  await expect(page.getByRole("heading", { name: "Sessions" })).toBeVisible();

  // The cookie is httpOnly: page scripts can't read it
  expect(await page.evaluate(() => document.cookie)).not.toContain("ss_session");

  await page.getByRole("button", { name: "Log out" }).filter({ visible: true }).first().click();
  await expect(page).toHaveURL(/\/login$/);
  await page.goto("/students");
  await expect(page).toHaveURL(/\/login$/);
});

test("a proctor reviews and approves a pending registration", async ({ page }) => {
  await login(page);
  await page.getByRole("link", { name: /students/i }).click();

  const card = page.locator("section").filter({ hasText: "Bilal Khan" });
  await expect(card).toBeVisible();
  const photo = card.getByRole("button", { name: /front photo/i });
  await expect(photo.locator("img")).toHaveJSProperty("complete", true);
  expect(await photo.locator("img").evaluate((img: HTMLImageElement) => img.naturalWidth)).toBe(320);

  await card.getByRole("button", { name: "Approve" }).click();
  await expect(page.getByText("Nothing to review")).toBeVisible();
  await page.getByRole("tab", { name: /approved/i }).click();
  await expect(page.locator("section").filter({ hasText: "Bilal Khan" })).toContainText(`Approved by ${E2E_USER.username}`);
});

test("run a session on a recorded video: live video, alerts, stop, report and evidence", async ({ page }) => {
  test.skip(!process.env.E2E_VIDEO_DIR, "set E2E_VIDEO_DIR to a folder with a test video");
  await login(page);

  await page.getByRole("link", { name: /new session/i }).click();
  await page.getByRole("checkbox", { name: /asha rao/i }).check();
  await page.locator("label", { hasText: /^book$/ }).click(); // the "book" allowed-device chip
  await expect(page.getByRole("checkbox", { name: "book", exact: true })).toBeChecked();
  await page.getByRole("radio").nth(1).check(); // the first video in E2E_VIDEO_DIR
  await page.getByRole("button", { name: /start session \(1\)/i }).click();

  await expect(page).toHaveURL(/\/live$/);
  const video = page.getByAltText("Live annotated camera feed");
  await expect(video).toBeVisible({ timeout: 60_000 }); // models load first
  await expect.poll(() => video.evaluate((img: HTMLImageElement) => img.naturalWidth), { timeout: 15_000 }).toBeGreaterThan(0);
  await expect(page.getByText(/\d+(\.\d)? fps/)).toBeVisible();
  await expect(page.getByText("Asha Rao")).toBeVisible();

  // The video shows someone who isn't registered: an alert with evidence arrives over the socket
  const firstAlert = page.getByRole("button", { name: /view evidence/i }).first();
  await expect(firstAlert).toBeVisible({ timeout: 45_000 });
  await firstAlert.click();
  const evidence = page.getByRole("dialog").getByRole("img");
  await expect.poll(() => evidence.evaluate((img: HTMLImageElement) => img.naturalWidth)).toBeGreaterThan(0);
  await page.keyboard.press("Escape");

  await page.getByRole("button", { name: "Stop session" }).click();
  await page.getByRole("dialog").getByRole("button", { name: "Stop session" }).click();
  await expect(page.getByText(/no exam session is running/i)).toBeVisible({ timeout: 30_000 });

  await page.getByRole("link", { name: "View report" }).click();
  await expect(page.getByRole("heading", { name: /session [0-9a-f]{8}/i })).toBeVisible();
  await expect(page.getByText("Asha Rao").first()).toBeVisible();
  await expect(page.getByText("Allowed: book")).toBeVisible();
  const thumb = page.getByRole("button", { name: /view evidence/i }).first().locator("img");
  await expect.poll(() => thumb.evaluate((img: HTMLImageElement) => img.naturalWidth)).toBeGreaterThan(0);

  await page.getByRole("link", { name: "All sessions" }).click();
  await expect(page.getByRole("row")).toHaveCount(2); // header + this session
});
