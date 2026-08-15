import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { TeamSelector } from "@/components/predict/TeamSelector";
import type { TeamInfo } from "@/lib/api/types";
import { mockFetchRoutes } from "@/test/helpers/mockFetch";
import teamsFixture from "@/test/fixtures/teams-search.json";

const vitality: TeamInfo = teamsFixture.teams[0] as TeamInfo;

function teamsBody(teams: Partial<TeamInfo>[]) {
  const full = teams.map((t) => ({
    canonical_name: "X",
    identity_eligible: true,
    history_available: true,
    history_match_count: 10,
    cold_start: false,
    ...t,
  }));
  return { request_id: "r", context_id: "deployment_post_cologne_v1", count: full.length, teams: full };
}

describe("TeamSelector", () => {
  it("searches the server (debounced) and renders canonical names with history metadata", async () => {
    mockFetchRoutes([{ match: "/api/v1/teams", body: teamsFixture }]);
    const onSelect = vi.fn();
    render(
      <TeamSelector label="Team A" side="a" selected={null} onSelect={onSelect} />,
    );
    await userEvent.type(screen.getByRole("combobox"), "vit");
    const option = await screen.findByRole("option", { name: /Team Vitality/ });
    expect(option).toHaveTextContent("196 matches");
  });

  it("selects via keyboard (ArrowDown + Enter) — full combobox behavior", async () => {
    mockFetchRoutes([{ match: "/api/v1/teams", body: teamsFixture }]);
    const onSelect = vi.fn();
    render(
      <TeamSelector label="Team A" side="a" selected={null} onSelect={onSelect} />,
    );
    const input = screen.getByRole("combobox");
    await userEvent.type(input, "vit");
    await screen.findByRole("option", { name: /Team Vitality/ });
    expect(input).toHaveAttribute("aria-expanded", "true");
    await userEvent.keyboard("{Enter}");
    expect(onSelect).toHaveBeenCalledWith(
      expect.objectContaining({ canonical_name: "Team Vitality" }),
    );
  });

  it("closes the listbox on Escape", async () => {
    mockFetchRoutes([{ match: "/api/v1/teams", body: teamsFixture }]);
    render(
      <TeamSelector label="Team A" side="a" selected={null} onSelect={vi.fn()} />,
    );
    const input = screen.getByRole("combobox");
    await userEvent.type(input, "vit");
    await screen.findByRole("option", { name: /Team Vitality/ });
    await userEvent.keyboard("{Escape}");
    expect(input).toHaveAttribute("aria-expanded", "false");
  });

  it("a slow older response can never overwrite a newer query's results (amendment #10)", async () => {
    // First query resolves SLOWLY with 'Old Team'; second resolves fast.
    mockFetchRoutes([
      { match: "q=vi", body: teamsBody([{ canonical_name: "Old Team" }]), delayMs: 300 },
      { match: "q=vit", body: teamsBody([{ canonical_name: "New Team" }]) },
    ]);
    render(
      <TeamSelector label="Team A" side="a" selected={null} onSelect={vi.fn()} />,
    );
    const input = screen.getByRole("combobox");
    // Type "vi", wait past the debounce so the slow request fires, then "t".
    await userEvent.type(input, "vi");
    await new Promise((r) => setTimeout(r, 270));
    await userEvent.type(input, "t");
    await screen.findByRole("option", { name: /New Team/ });
    // Give the slow stale response time to arrive — it must be discarded.
    await new Promise((r) => setTimeout(r, 350));
    expect(screen.queryByRole("option", { name: /Old Team/ })).toBeNull();
    expect(screen.getByRole("option", { name: /New Team/ })).toBeInTheDocument();
  });

  it("does not hit the server for an empty query", async () => {
    const fetchMock = mockFetchRoutes([
      { match: "/api/v1/teams", body: teamsFixture },
    ]);
    render(
      <TeamSelector label="Team A" side="a" selected={null} onSelect={vi.fn()} />,
    );
    screen.getByRole("combobox").focus();
    await new Promise((r) => setTimeout(r, 320));
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("marks the team already selected on the other side as not selectable", async () => {
    mockFetchRoutes([{ match: "/api/v1/teams", body: teamsFixture }]);
    const onSelect = vi.fn();
    render(
      <TeamSelector
        label="Team B"
        side="b"
        selected={null}
        onSelect={onSelect}
        excludeName="Team Vitality"
      />,
    );
    await userEvent.type(screen.getByRole("combobox"), "vit");
    const option = await screen.findByRole("option", { name: /Team Vitality/ });
    expect(option).toHaveAttribute("aria-disabled", "true");
    expect(option).toHaveTextContent("already selected");
    await userEvent.pointer({ keys: "[MouseLeft]", target: option });
    expect(onSelect).not.toHaveBeenCalled();
  });

  it("shows the selected identity panel with history metadata and a clear affordance", async () => {
    const onSelect = vi.fn();
    render(
      <TeamSelector label="Team A" side="a" selected={vitality} onSelect={onSelect} />,
    );
    expect(screen.getByText("Team Vitality")).toBeInTheDocument();
    expect(screen.getByText(/196 matches in snapshot/)).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: /Clear Team A/ }));
    expect(onSelect).toHaveBeenCalledWith(null);
  });

  it("labels cold-start teams as having no snapshot history", () => {
    render(
      <TeamSelector
        label="Team A"
        side="a"
        selected={{ ...vitality, canonical_name: "Team Germany", cold_start: true, history_match_count: 0 }}
        onSelect={vi.fn()}
      />,
    );
    expect(screen.getByText(/No match history in this snapshot/)).toBeInTheDocument();
  });
});
