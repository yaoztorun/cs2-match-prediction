import { describe, expect, it } from "vitest";
import { formatDataThrough, formatPercent } from "@/lib/format";

describe("formatPercent", () => {
  it("rounds to whole percent by default", () => {
    expect(formatPercent(0.5523408433787343)).toBe("55%");
    expect(formatPercent(0.44765915662126565)).toBe("45%");
  });
  it("supports decimals", () => {
    expect(formatPercent(0.5523408433787343, 1)).toBe("55.2%");
  });
});

describe("formatDataThrough (amendment #20: no timezone conversion)", () => {
  it("formats the ISO variant returned by /meta", () => {
    expect(formatDataThrough("2026-06-28T20:00:00")).toBe("28 Jun 2026");
  });
  it("formats the space-separated variant returned by prediction metadata", () => {
    expect(formatDataThrough("2026-06-28 20:00:00")).toBe("28 Jun 2026");
  });
  it("reads date components from the string — a late-evening timestamp can never shift calendar date", () => {
    // new Date("2026-06-28T20:00:00Z") in a UTC+X zone could become 29 Jun;
    // string-level parsing makes that impossible.
    expect(formatDataThrough("2026-06-28T23:59:59")).toBe("28 Jun 2026");
  });
  it("passes through unparseable strings unchanged", () => {
    expect(formatDataThrough("unknown")).toBe("unknown");
  });
});
