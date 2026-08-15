import type { Metadata } from "next";
import { PredictClient } from "@/components/predict/PredictClient";

export const metadata: Metadata = {
  title: "Predict",
  description:
    "Model-grounded CS2 series probabilities — pre-veto or with the exact ordered maps.",
};

export default function PredictPage() {
  return <PredictClient />;
}
