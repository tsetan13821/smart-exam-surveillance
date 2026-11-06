import { describe, expect, it } from "vitest";

import { alertLabel, eventTone, examRules, formatDuration, isIncident, secondsBetween, sourceLabel } from "./format";

describe("alertLabel", () => {
  it("uses friendly names for known alert types", () => {
    expect(alertLabel("HEAD_TURN_ALERT")).toBe("Head turned away");
    expect(alertLabel("CONTRABAND_DETECTED")).toBe("Prohibited device");
  });
  it("derives a readable label for unknown types", () => {
    expect(alertLabel("SOMETHING_NEW_ALERT")).toBe("Something new");
    expect(alertLabel(null)).toBe("Event");
  });
});

describe("incidents and tones", () => {
  it("does not count attendance and breaks as incidents", () => {
    expect(isIncident("EVIDENCE", "ATTENDANCE_MARKED")).toBe(false);
    expect(isIncident("EVIDENCE", "PRESENCE_LEFT")).toBe(false);
    expect(isIncident("EVIDENCE", "STUDENT_MISSING")).toBe(true);
    expect(isIncident("ALERT", "CAMERA_ERROR")).toBe(false);
  });
  it("colours serious events red", () => {
    expect(eventTone("EVIDENCE", "CONTRABAND_DETECTED")).toBe("danger");
    expect(eventTone("EVIDENCE", "HEAD_TURN_ALERT")).toBe("warning");
    expect(eventTone("EVIDENCE", "ATTENDANCE_MARKED")).toBe("success");
    expect(eventTone("ALERT", "CAMERA_ERROR")).toBe("danger");
  });
});

describe("formatDuration", () => {
  it.each([
    [0, "0s"],
    [59.6, "1m 00s"],
    [75, "1m 15s"],
    [3700, "1h 1m"],
    [null, "—"],
    [Number.NaN, "—"],
  ])("%s -> %s", (input, expected) => {
    expect(formatDuration(input)).toBe(expected);
  });
});

describe("secondsBetween", () => {
  it("measures to now when the end is missing and never goes negative", () => {
    const start = "2026-11-06T10:00:00";
    expect(secondsBetween(start, "2026-11-06T10:01:30")).toBe(90);
    expect(secondsBetween(start, null, new Date("2026-11-06T10:00:10").getTime())).toBe(10);
    expect(secondsBetween(start, "2026-11-06T09:00:00")).toBe(0);
  });
});

describe("sourceLabel and examRules", () => {
  it("shows only the file name of a video, never its folder", () => {
    expect(sourceLabel("C:\\data\\demo_videos\\exam.mp4")).toBe("Video: exam.mp4");
    expect(sourceLabel("/srv/demo_videos/exam.mp4")).toBe("Video: exam.mp4");
    expect(sourceLabel("0")).toBe("Camera 0");
    expect(sourceLabel(null)).toBe("Default camera");
  });
  it("summarises the rules", () => {
    expect(examRules({ is_open_book: false, is_paper_exam: true, allowed_devices: ["book"] })).toEqual([
      "Closed book",
      "Paper exam",
      "Allowed: book",
    ]);
  });
});
