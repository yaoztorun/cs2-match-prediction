/**
 * Opt-in integration smoke test against the REAL local FastAPI service.
 * Skipped unless RUN_API_SMOKE=1, so the normal frontend suite never
 * depends on Python server availability (spec §40).
 *
 *   RUN_API_SMOKE=1 npm run test:smoke
 */
import { describe, expect, it } from "vitest";

const RUN = process.env.RUN_API_SMOKE === "1";
const BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

describe.skipIf(!RUN)("live API smoke", () => {
  it("GET /meta matches the frozen contract", async () => {
    const res = await fetch(`${BASE}/api/v1/meta`);
    expect(res.ok).toBe(true);
    const meta = await res.json();
    expect(meta.default_context_id).toBe("deployment_post_cologne_v1");
    expect(meta.deployment_state_data_through).toMatch(/^2026-06-28/);
    expect(meta.state_is_live).toBe(false);
  });

  it("POST /predict/series (pre-veto) returns the envelope shape the UI renders", async () => {
    const res = await fetch(`${BASE}/api/v1/predict/series`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        context_id: "deployment_post_cologne_v1",
        mode: "pre_veto",
        team_a: "Team Vitality",
        team_b: "Team Falcons",
        best_of: 3,
        include_explanation: true,
        explanation_detail: "summary",
      }),
    });
    expect(res.ok).toBe(true);
    const body = await res.json();
    expect(Object.keys(body).sort()).toEqual([
      "api_version",
      "explanation",
      "metadata",
      "prediction",
      "request_id",
    ]);
    expect(body.prediction.probability_team_a).toBeGreaterThan(0);
    expect(body.prediction.probability_team_a).toBeLessThan(1);
    expect(body.explanation.detail_level).toBe("summary");
    expect(body.explanation.causal).toBe(false);
  });
});
