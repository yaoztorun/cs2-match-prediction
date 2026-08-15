import type { Metadata } from "next";
import { ComingNextPhase } from "@/components/shell/ComingNextPhase";

export const metadata: Metadata = { title: "Cologne 2026 Results" };

export default function CologneResultsPage() {
  return (
    <ComingNextPhase
      title="Cologne 2026 — simulation vs reality"
      description="How the frozen pre-event simulation compared with what actually happened at the Major — match accuracy, milestone overlap, and the champion the model didn't see coming."
      upcoming={[
        "Actual champion vs pre-event odds and ranking",
        "Match-level accuracy and calibration metrics",
        "Predicted vs actual playoff and semifinal fields",
      ]}
    />
  );
}
