import "@testing-library/jest-dom/vitest";
import { afterEach, vi } from "vitest";
import { cleanup } from "@testing-library/react";
import { __resetMetaCache } from "@/lib/hooks/useMeta";
import { __resetMapsCache } from "@/components/predict/OrderedMapSelector";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  __resetMetaCache();
  __resetMapsCache();
  // PredictClient syncs inputs into the query string via replaceState —
  // reset so no test inherits another test's URL state.
  window.history.replaceState(null, "", "/");
});
