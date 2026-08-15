import { describe, expect, it } from "vitest";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { PredictionResult } from "@/components/predict/PredictionResult";
import type {
  KnownMapsEnvelope,
  PreVetoEnvelope,
} from "@/lib/api/types";
import preveto from "@/test/fixtures/predict-preveto-bo3.json";
import knownBo3 from "@/test/fixtures/predict-knownmaps-bo3.json";
import coldStart from "@/test/fixtures/predict-coldstart.json";
import partialSupport from "@/test/fixtures/predict-partial-support.json";

const prevetoEnvelope = preveto as unknown as PreVetoEnvelope;
const knownBo3Envelope = knownBo3 as unknown as KnownMapsEnvelope;
const coldStartEnvelope = coldStart as unknown as PreVetoEnvelope;
const partialEnvelope = partialSupport as unknown as KnownMapsEnvelope;

describe("PredictionResult — pre-veto (real captured fixture)", () => {
  it("renders both teams, rounded percentages, and the favored team", () => {
    render(<PredictionResult envelope={prevetoEnvelope} />);
    expect(screen.getAllByText("Team Vitality").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Team Falcons").length).toBeGreaterThan(0);
    expect(screen.getByText("55%")).toBeInTheDocument(); // 0.5523…
    expect(screen.getByText("45%")).toBeInTheDocument();
    expect(screen.getByText(/Favored:/)).toBeInTheDocument();
  });

  it("labels the mode as pre-veto without surfacing model jargon in the hero", () => {
    render(<PredictionResult envelope={prevetoEnvelope} />);
    expect(screen.getByText(/Pre-veto prediction/)).toBeInTheDocument();
    const hero = screen.getByRole("region", { name: "Prediction result" });
    expect(within(hero).queryByText(/random_forest|Random Forest/)).toBeNull();
  });

  it("renders only the factor groups the API returned, split by team", () => {
    render(<PredictionResult envelope={prevetoEnvelope} />);
    // RF pre-veto exposes only the four RF concepts — fixture-driven.
    expect(screen.getByText("Overall strength")).toBeInTheDocument();
    expect(screen.getByText("Recent performance")).toBeInTheDocument();
    // XGB-only concepts must NOT appear for an RF prediction.
    expect(screen.queryByText("Selected-map strength")).toBeNull();
    expect(screen.queryByText("Map-pool depth")).toBeNull();
    expect(screen.getByText(/Supporting Team Vitality/)).toBeInTheDocument();
    expect(screen.getByText(/Supporting Team Falcons/)).toBeInTheDocument();
  });

  it("includes the non-causal disclosure once", () => {
    render(<PredictionResult envelope={prevetoEnvelope} />);
    expect(
      screen.getAllByText(/do not prove causal reasons/),
    ).toHaveLength(1);
  });

  it("shows the freshness date derived from response metadata", () => {
    render(<PredictionResult envelope={prevetoEnvelope} />);
    expect(screen.getByText("28 Jun 2026")).toBeInTheDocument();
  });

  it("details drawer names the RF method as Saabas-style, never SHAP", async () => {
    render(<PredictionResult envelope={prevetoEnvelope} />);
    await userEvent.click(
      screen.getByRole("button", { name: /Prediction details/ }),
    );
    expect(
      screen.getByText("Saabas-style tree path decomposition"),
    ).toBeInTheDocument();
    expect(screen.queryByText(/^SHAP/)).toBeNull();
  });

  it("never renders a % label on factor contribution bars (amendment #7)", () => {
    const { container } = render(<PredictionResult envelope={prevetoEnvelope} />);
    const bars = container.querySelectorAll(
      '[title="Relative model contribution (within this prediction only)"]',
    );
    expect(bars.length).toBeGreaterThan(0);
    for (const bar of bars) {
      expect(bar.textContent).not.toContain("%");
    }
  });
});

describe("PredictionResult — known maps BO3 (real captured fixture)", () => {
  it("renders the composed series probability and every map row", () => {
    render(<PredictionResult envelope={knownBo3Envelope} />);
    expect(screen.getByText(/Known-map series prediction/)).toBeInTheDocument();
    expect(screen.getByText("63%")).toBeInTheDocument(); // 0.6327…
    expect(screen.getByText("Map 1")).toBeInTheDocument();
    expect(screen.getByText("Map 2")).toBeInTheDocument();
    expect(screen.getByText("Map 3")).toBeInTheDocument();
    expect(screen.getByText("Mirage")).toBeInTheDocument();
    expect(screen.getByText("Inferno")).toBeInTheDocument();
    expect(screen.getByText("Nuke")).toBeInTheDocument();
  });

  it("shows map reach probability from series_composition (amendment: never hidden)", () => {
    render(<PredictionResult envelope={knownBo3Envelope} />);
    const reachLabels = screen.getAllByText(/Chance map is played:/);
    expect(reachLabels).toHaveLength(3);
    // Map 3 reach = 0.4841993… -> 48%
    expect(screen.getByText("48%")).toBeInTheDocument();
  });

  it("keeps series mechanics visually separate from model factors (amendment #9)", () => {
    render(<PredictionResult envelope={knownBo3Envelope} />);
    // Leverage is rendered as series metadata, present for each map…
    expect(screen.getAllByText(/Series leverage:/)).toHaveLength(3);
    // …and per-map factors are collapsed until explicitly expanded.
    expect(screen.queryByText(/Why the model leans this way on Mirage/)).toBeNull();
  });

  it("expands per-map TreeSHAP factors on demand, labeled for that one map (amendment #8)", async () => {
    render(<PredictionResult envelope={knownBo3Envelope} />);
    await userEvent.click(
      screen.getByRole("button", { name: /Show factors for Mirage/ }),
    );
    expect(
      screen.getByText("Why the model leans this way on Mirage"),
    ).toBeInTheDocument();
    // No merged "series factor ranking" section exists.
    expect(
      screen.getByText(/never merged into a single series factor ranking/),
    ).toBeInTheDocument();
  });
});

describe("PredictionResult — data support notices (real captured fixtures)", () => {
  it("surfaces a cold-start team politely, separate from model factors", () => {
    render(<PredictionResult envelope={coldStartEnvelope} />);
    const aside = screen.getByRole("complementary", { name: "Data coverage" });
    expect(aside).toHaveTextContent(
      /Team Germany has no match history in this data snapshot/,
    );
    expect(aside.textContent?.toLowerCase()).not.toContain("weak");
  });

  it("describes known-map fallbacks_used as limited data, never a team weakness", () => {
    render(<PredictionResult envelope={partialEnvelope} />);
    const aside = screen.getByRole("complementary", { name: "Data coverage" });
    expect(aside).toHaveTextContent(/map-specific history/);
    expect(aside).toHaveTextContent(/neutral defaults/);
    expect(aside.textContent?.toLowerCase()).not.toContain("weak");
  });

  it("shows no notice for a fully supported matchup", () => {
    render(<PredictionResult envelope={prevetoEnvelope} />);
    expect(
      screen.queryByRole("complementary", { name: "Data coverage" }),
    ).toBeNull();
  });
});
