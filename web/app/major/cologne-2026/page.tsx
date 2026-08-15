import type { Metadata } from "next";
import { ComingNextPhase } from "@/components/shell/ComingNextPhase";

export const metadata: Metadata = { title: "Cologne 2026" };

export default function ColognePreEventPage() {
  return (
    <ComingNextPhase
      title="Cologne 2026 — pre-event simulation"
      description="The frozen 50,000-run simulation computed before IEM Cologne Major 2026 began: championship odds, stage advancement, and the model's favorite path."
      upcoming={[
        "Pre-event championship probabilities for all 32 teams",
        "Swiss record and playoff seed distributions",
        "The deterministic favorite-wins bracket",
      ]}
    />
  );
}
