import { describe, expect, it, vi } from "vitest";
import { ApiError, ApiUnreachableError, apiGet, apiPost } from "@/lib/api/client";
import { predictSeries, searchTeams } from "@/lib/api/endpoints";
import { mockFetchRoutes } from "@/test/helpers/mockFetch";
import errorFixture from "@/test/fixtures/error-unknown-team.json";
import teamsFixture from "@/test/fixtures/teams-search.json";
import prevetoFixture from "@/test/fixtures/predict-preveto-bo3.json";

describe("api client", () => {
  it("performs typed GETs with query params", async () => {
    const fetchMock = mockFetchRoutes([
      { match: "/api/v1/teams", body: teamsFixture },
    ]);
    const res = await searchTeams("vit");
    expect(res.teams[0].canonical_name).toBe("Team Vitality");
    const calledUrl = String(fetchMock.mock.calls[0][0]);
    expect(calledUrl).toContain("/api/v1/teams");
    expect(calledUrl).toContain("q=vit");
    expect(calledUrl).toContain("context_id=deployment_post_cologne_v1");
  });

  it("parses the structured error envelope into a typed ApiError", async () => {
    mockFetchRoutes([
      { match: "/api/v1/predict/series", body: errorFixture, status: 404 },
    ]);
    const err = await predictSeries({
      teamA: "Not A Real Team",
      teamB: "Team Vitality",
      bestOf: 3,
      mode: "pre_veto",
    }).catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.code).toBe("unknown_team");
    expect(err.status).toBe(404);
    expect(err.requestId).toBe(errorFixture.request_id);
  });

  it("wraps network failures as ApiUnreachableError", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        throw new TypeError("Failed to fetch");
      }),
    );
    const err = await apiGet("/api/v1/meta").catch((e) => e);
    expect(err).toBeInstanceOf(ApiUnreachableError);
  });

  it("sends the frozen prediction contract: summary explanation, no prediction_datetime", async () => {
    const fetchMock = mockFetchRoutes([
      { match: "/api/v1/predict/series", body: prevetoFixture },
    ]);
    await predictSeries({
      teamA: "Team Vitality",
      teamB: "Team Falcons",
      bestOf: 3,
      mode: "pre_veto",
    });
    const body = JSON.parse(String(fetchMock.mock.calls[0][1]?.body));
    expect(body).toEqual({
      context_id: "deployment_post_cologne_v1",
      mode: "pre_veto",
      team_a: "Team Vitality",
      team_b: "Team Falcons",
      best_of: 3,
      include_explanation: true,
      explanation_detail: "summary",
    });
    expect(body).not.toHaveProperty("prediction_datetime");
    expect(body).not.toHaveProperty("ordered_maps");
  });

  it("includes ordered_maps only in known_maps mode, preserving order", async () => {
    const fetchMock = mockFetchRoutes([
      { match: "/api/v1/predict/series", body: prevetoFixture },
    ]);
    await predictSeries({
      teamA: "Team Vitality",
      teamB: "Team Falcons",
      bestOf: 3,
      mode: "known_maps",
      orderedMaps: ["Mirage", "Nuke", "Inferno"],
    });
    const body = JSON.parse(String(fetchMock.mock.calls[0][1]?.body));
    expect(body.ordered_maps).toEqual(["Mirage", "Nuke", "Inferno"]);
  });

  it("apiPost surfaces non-JSON error bodies as generic http_error", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response("<html>bad gateway</html>", { status: 502 })),
    );
    const err = (await apiPost("/api/v1/predict/series", {}).catch(
      (e) => e,
    )) as ApiError;
    expect(err).toBeInstanceOf(ApiError);
    expect(err.code).toBe("http_error");
  });
});
