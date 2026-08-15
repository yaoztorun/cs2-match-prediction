import { describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { BestOfSelector } from "@/components/predict/BestOfSelector";
import { MapModeSelector } from "@/components/predict/MapModeSelector";
import { OrderedMapSelector } from "@/components/predict/OrderedMapSelector";
import { mockFetchRoutes } from "@/test/helpers/mockFetch";
import mapsFixture from "@/test/fixtures/maps.json";

describe("BestOfSelector", () => {
  it("offers exactly BO1/BO3/BO5 as a segmented radio group with typed values", async () => {
    const onChange = vi.fn();
    render(<BestOfSelector value={3} onChange={onChange} />);
    const radios = screen.getAllByRole("radio");
    expect(radios.map((r) => r.textContent)).toEqual(["BO1", "BO3", "BO5"]);
    expect(screen.getByRole("radio", { name: "BO3" })).toHaveAttribute(
      "aria-checked",
      "true",
    );
    await userEvent.click(screen.getByRole("radio", { name: "BO5" }));
    expect(onChange).toHaveBeenCalledWith(5); // typed int, never "5"
  });
});

describe("MapModeSelector", () => {
  it("presents the two modes with product wording, not model jargon", () => {
    render(<MapModeSelector value="pre_veto" onChange={vi.fn()} />);
    expect(screen.getByText("Maps unknown")).toBeInTheDocument();
    expect(screen.getByText("Before veto / map selection")).toBeInTheDocument();
    expect(screen.getByText("Maps known")).toBeInTheDocument();
    expect(screen.getByText("Use the exact ordered maps")).toBeInTheDocument();
    expect(screen.queryByText(/Random Forest|XGBoost/)).toBeNull();
  });
  it("switches modes", async () => {
    const onChange = vi.fn();
    render(<MapModeSelector value="pre_veto" onChange={onChange} />);
    await userEvent.click(screen.getByRole("radio", { name: /Maps known/ }));
    expect(onChange).toHaveBeenCalledWith("known_maps");
  });
});

describe("OrderedMapSelector", () => {
  it.each([
    [1, 1],
    [3, 3],
    [5, 5],
  ] as const)("renders exactly %i slot(s) for BO%i", async (bo, expected) => {
    mockFetchRoutes([{ match: "/api/v1/maps", body: mapsFixture }]);
    render(<OrderedMapSelector bestOf={bo} value={[]} onChange={vi.fn()} />);
    expect(screen.getAllByRole("combobox")).toHaveLength(expected);
    await waitFor(() =>
      expect(screen.getAllByRole("combobox")[0]).not.toBeDisabled(),
    );
  });

  it("offers only maps the API marks model_supported (amendment #16) — Cache never appears", async () => {
    const withUnsupported = {
      ...mapsFixture,
      maps: [
        ...mapsFixture.maps,
        {
          map_name: "Cache",
          model_supported: false,
          historical_context_available: false,
          deployment_context_observed: false,
          cologne_2026_competitive_pool: false,
          competitive_pool_status: "unsupported",
        },
      ],
    };
    mockFetchRoutes([{ match: "/api/v1/maps", body: withUnsupported }]);
    render(<OrderedMapSelector bestOf={1} value={[]} onChange={vi.fn()} />);
    await waitFor(() =>
      expect(screen.getAllByRole("option", { name: "Mirage" }).length).toBe(1),
    );
    expect(screen.queryByRole("option", { name: /Cache/ })).toBeNull();
    // The 9 supported maps + the empty "Select…" option
    expect(screen.getAllByRole("option")).toHaveLength(10);
  });

  it("disables maps already picked in another slot (duplicate prevention)", async () => {
    mockFetchRoutes([{ match: "/api/v1/maps", body: mapsFixture }]);
    render(
      <OrderedMapSelector
        bestOf={3}
        value={["Mirage", "", ""]}
        onChange={vi.fn()}
      />,
    );
    await waitFor(() =>
      expect(screen.getAllByRole("combobox")[1]).not.toBeDisabled(),
    );
    const slot2 = screen.getAllByRole("combobox")[1];
    const mirageInSlot2 = Array.from(slot2.querySelectorAll("option")).find(
      (o) => o.value === "Mirage",
    );
    expect(mirageInSlot2?.disabled).toBe(true);
    expect(mirageInSlot2?.textContent).toContain("already picked");
  });

  it('is labeled "supported maps", never "map pool"/"Active Duty" (amendment #17)', async () => {
    mockFetchRoutes([{ match: "/api/v1/maps", body: mapsFixture }]);
    const { container } = render(
      <OrderedMapSelector bestOf={1} value={[]} onChange={vi.fn()} />,
    );
    expect(container.textContent).toContain("supported maps");
    expect(container.textContent?.toLowerCase()).not.toContain("map pool");
    expect(container.textContent?.toLowerCase()).not.toContain("active duty");
  });
});
