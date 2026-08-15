import type { Metadata } from "next";
import { ComingNextPhase } from "@/components/shell/ComingNextPhase";

export const metadata: Metadata = { title: "Major" };

export default function MajorPage() {
  return (
    <ComingNextPhase
      title="Major simulator"
      description="Simulate a full 32-team Major — Swiss stages, playoffs, and championship odds — powered by the same frozen prediction model."
      upcoming={[
        "Interactive Pick'Em: override any match and watch the bracket recompute",
        "Monte Carlo championship odds for a custom 32-team field",
        "The frozen pre-event Cologne 2026 simulation and how it compared with reality",
      ]}
    />
  );
}
