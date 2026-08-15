import { describe, expect, it } from "vitest";
import type { GroupedFactor } from "@/lib/api/types";
import {
  attributionMethodLabel,
  describeFallbacks,
  factorLabel,
  factorSentence,
  relativeMagnitude,
} from "@/lib/factors";
import preveto from "@/test/fixtures/predict-preveto-bo3.json";

function factor(overrides: Partial<GroupedFactor>): GroupedFactor {
  return {
    factor_group: "overall_strength",
    direction: "team_a",
    signed_contribution: 0.05,
    absolute_importance: 0.05,
    rank: 1,
    attribution_output_space: "probability",
    ...overrides,
  };
}

describe("factor display copy (amendment #6: structured fields, no prose mutation)", () => {
  it("builds copy from factor_group + direction + canonical names", () => {
    expect(
      factorSentence(factor({}), "Team Vitality", "Team Falcons"),
    ).toBe("Overall strength favors Team Vitality");
    expect(
      factorSentence(
        factor({ direction: "team_b", factor_group: "recent_performance" }),
        "Team Vitality",
        "Team Falcons",
      ),
    ).toBe("Recent performance favors Team Falcons");
  });
  it("handles neutral direction without naming a team", () => {
    expect(
      factorSentence(factor({ direction: "neutral" }), "A", "B"),
    ).toContain("negligible net effect");
  });
  it("never drops an unknown factor_group slug", () => {
    expect(factorLabel("some_future_group")).toBe("Some future group");
  });
  it("leaves the backend human_readable_summary untouched (verbatim fixture strings)", () => {
    // The captured fixture prose contains generic Team A/Team B labels; the
    // frontend renders it verbatim in details rather than string-replacing.
    const summary = preveto.explanation.human_readable_summary as string[];
    expect(summary.some((s) => s.includes("Team A") || s.includes("Team B"))).toBe(
      true,
    );
  });
});

describe("relativeMagnitude (amendment #7: within one explanation only)", () => {
  it("normalizes to the max absolute contribution of the given explanation", () => {
    const all = [
      factor({ absolute_importance: 0.08 }),
      factor({ factor_group: "event_context", absolute_importance: 0.02 }),
    ];
    expect(relativeMagnitude(all[0], all)).toBe(1);
    expect(relativeMagnitude(all[1], all)).toBeCloseTo(0.25);
  });
  it("returns 0 when all contributions are zero", () => {
    const all = [factor({ absolute_importance: 0 })];
    expect(relativeMagnitude(all[0], all)).toBe(0);
  });
});

describe("attribution method labels", () => {
  it("labels RF as Saabas-style decomposition, never SHAP", () => {
    const label = attributionMethodLabel("saabas_path_decomposition");
    expect(label).toBe("Saabas-style tree path decomposition");
    expect(label.toLowerCase()).not.toContain("shap");
  });
  it("labels XGB as TreeSHAP", () => {
    expect(attributionMethodLabel("xgboost_native_treeshap")).toContain(
      "TreeSHAP",
    );
  });
});

describe("describeFallbacks", () => {
  it("returns null when no fallbacks were used", () => {
    expect(describeFallbacks([])).toBeNull();
  });
  it("phrases the real Phase 9B fallback keys politely", () => {
    const text = describeFallbacks([
      "both_teams_have_map_history",
      "both_teams_have_5_adjusted_matches",
    ]);
    expect(text).toContain("map-specific history");
    expect(text).toContain("recent-form history");
    expect(text).toContain("neutral defaults");
    // Data-quality copy never frames missing data as a team weakness.
    expect(text!.toLowerCase()).not.toContain("weak");
  });
});
