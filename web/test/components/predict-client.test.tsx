import { describe, expect, it } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { PredictClient } from "@/components/predict/PredictClient";
import { mockFetchRoutes, type FetchRoute } from "@/test/helpers/mockFetch";
import teamsFixture from "@/test/fixtures/teams-search.json";
import mapsFixture from "@/test/fixtures/maps.json";
import metaFixture from "@/test/fixtures/meta.json";
import healthFixture from "@/test/fixtures/health-ready.json";
import prevetoFixture from "@/test/fixtures/predict-preveto-bo3.json";
import errorFixture from "@/test/fixtures/error-unknown-team.json";

const falconsTeams = {
  request_id: "r",
  context_id: "deployment_post_cologne_v1",
  count: 1,
  teams: [
    {
      canonical_name: "Team Falcons",
      identity_eligible: true,
      history_available: true,
      history_match_count: 199,
      cold_start: false,
    },
  ],
};

function baseRoutes(extra: FetchRoute[] = []): FetchRoute[] {
  return [
    { match: "/api/v1/health/ready", body: healthFixture },
    { match: "/api/v1/meta", body: metaFixture },
    { match: "/api/v1/maps", body: mapsFixture },
    { match: "/api/v1/teams", body: teamsFixture },
    { match: "q=fal", body: falconsTeams },
    { match: "/api/v1/predict/series", body: prevetoFixture },
    ...extra,
  ];
}

async function pickTeams() {
  const [inputA] = screen.getAllByRole("combobox");
  await userEvent.type(inputA, "vit");
  await userEvent.click(
    await screen.findByRole("option", { name: /Team Vitality/ }),
  );
  const inputB = screen.getByRole("combobox");
  await userEvent.type(inputB, "fal");
  await userEvent.click(
    await screen.findByRole("option", { name: /Team Falcons/ }),
  );
}

describe("PredictClient", () => {
  it("runs a pre-veto prediction end-to-end and renders the result", async () => {
    mockFetchRoutes(baseRoutes());
    render(<PredictClient />);
    const predict = screen.getByRole("button", { name: "Predict match" });
    expect(predict).toBeDisabled(); // incomplete inputs
    await pickTeams();
    await waitFor(() => expect(predict).toBeEnabled());
    await userEvent.click(predict);
    expect(await screen.findByText("55%")).toBeInTheDocument();
    expect(screen.getByText(/Favored:/)).toBeInTheDocument();
  });

  it("invalidates the displayed result when a prediction-defining input changes (amendment #12)", async () => {
    mockFetchRoutes(baseRoutes());
    render(<PredictClient />);
    await pickTeams();
    await userEvent.click(screen.getByRole("button", { name: "Predict match" }));
    await screen.findByText("55%");
    // BO3 -> BO5: the old BO3 result must not keep rendering.
    await userEvent.click(screen.getByRole("radio", { name: "BO5" }));
    expect(screen.queryByText("55%")).toBeNull();
    expect(
      screen.getByText(/Inputs changed — run Predict to update the result./),
    ).toBeInTheDocument();
  });

  it("keeps the result when inputs are changed back to the exact signature that produced it", async () => {
    mockFetchRoutes(baseRoutes());
    render(<PredictClient />);
    await pickTeams();
    await userEvent.click(screen.getByRole("button", { name: "Predict match" }));
    await screen.findByText("55%");
    await userEvent.click(screen.getByRole("radio", { name: "BO5" }));
    expect(screen.queryByText("55%")).toBeNull();
    await userEvent.click(screen.getByRole("radio", { name: "BO3" }));
    expect(screen.getByText("55%")).toBeInTheDocument();
  });

  it("requires the exact ordered map count in known-maps mode before enabling Predict", async () => {
    mockFetchRoutes(baseRoutes());
    render(<PredictClient />);
    await pickTeams();
    await userEvent.click(screen.getByRole("radio", { name: /Maps known/ }));
    const predict = screen.getByRole("button", { name: "Predict match" });
    expect(predict).toBeDisabled();
    const slots = () =>
      screen
        .getAllByRole("combobox")
        .filter((el) => el.tagName === "SELECT") as HTMLSelectElement[];
    await waitFor(() => expect(slots()).toHaveLength(3));
    await waitFor(() => expect(slots()[0]).toBeEnabled());
    await userEvent.selectOptions(slots()[0], "Mirage");
    await userEvent.selectOptions(slots()[1], "Inferno");
    expect(predict).toBeDisabled(); // 2 of 3 picked
    await userEvent.selectOptions(slots()[2], "Nuke");
    await waitFor(() => expect(predict).toBeEnabled());
  });

  it("maps a structured backend error to friendly copy (unknown_team)", async () => {
    mockFetchRoutes(
      baseRoutes([
        { match: "/api/v1/predict/series", body: errorFixture, status: 404 },
      ]),
    );
    render(<PredictClient />);
    await pickTeams();
    await userEvent.click(screen.getByRole("button", { name: "Predict match" }));
    expect(
      await screen.findByText(
        "That team isn't available in this model snapshot.",
      ),
    ).toBeInTheDocument();
    expect(screen.queryByText(/Phase 2.5 policy/)).toBeNull(); // no raw backend text
  });

  it("shows the polished unavailable state with retry when readiness fails", async () => {
    mockFetchRoutes([
      {
        match: "/api/v1/health/ready",
        body: { error: { code: "service_unavailable", message: "down", detail: {} }, request_id: "r" },
        status: 503,
      },
      { match: "/api/v1/meta", body: metaFixture },
    ]);
    render(<PredictClient />);
    expect(
      await screen.findByText("Prediction service unavailable"),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Retry/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Predict match" })).toBeDisabled();
  });

  it("restores complete URL state and auto-runs exactly once (amendment #13)", async () => {
    const fetchMock = mockFetchRoutes(
      baseRoutes([
        { match: "q=Team+Vitality", body: teamsFixture },
        { match: "q=Team+Falcons", body: falconsTeams },
      ]),
    );
    window.history.replaceState(
      null,
      "",
      "?a=Team%20Vitality&b=Team%20Falcons&bo=3&mode=pre_veto",
    );
    render(<PredictClient />);
    expect(await screen.findByText("55%")).toBeInTheDocument();
    const predictCalls = fetchMock.mock.calls.filter(([u]) =>
      String(u).includes("/predict/series"),
    );
    expect(predictCalls).toHaveLength(1); // exactly one auto-run
    window.history.replaceState(null, "", "/");
  });

  it("shows a clean unavailable message when a URL team is not in the snapshot (amendment #14)", async () => {
    mockFetchRoutes(
      baseRoutes([
        {
          match: "q=Ghost+Team",
          body: { request_id: "r", context_id: "c", count: 0, teams: [] },
        },
        { match: "q=Team+Falcons", body: falconsTeams },
      ]),
    );
    window.history.replaceState(
      null,
      "",
      "?a=Ghost%20Team&b=Team%20Falcons&bo=3&mode=pre_veto",
    );
    render(<PredictClient />);
    expect(
      await screen.findByText(/Ghost Team isn't available in this model snapshot/),
    ).toBeInTheDocument();
    window.history.replaceState(null, "", "/");
  });
});
