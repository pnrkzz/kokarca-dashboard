import { create } from "zustand";
import type { TimeLayer } from "../types";

type TimeMode = "all" | "single" | "range" | "cumulative" | "animation";
type AnimationKind = "annual" | "cumulative";
type ViewKey = "map" | "stc" | "climate" | "interventions" | "documents" | "questions" | "methodology";

type LinkedState = {
  activeView: ViewKey;
  selectedProvince: string | null;
  selectedProduct: string;
  timeMode: TimeMode;
  timeLayer: TimeLayer;
  animationKind: AnimationKind;
  selectedYear: number;
  rangeStart: number;
  rangeEnd: number;
  isPlaying: boolean;
  setActiveView: (view: ViewKey) => void;
  setSelectedProvince: (province: string | null) => void;
  setSelectedProduct: (product: string) => void;
  setTimeMode: (mode: TimeMode) => void;
  setTimeLayer: (layer: TimeLayer) => void;
  setAnimationKind: (kind: AnimationKind) => void;
  setSelectedYear: (year: number) => void;
  setRangeStart: (year: number) => void;
  setRangeEnd: (year: number) => void;
  setIsPlaying: (playing: boolean) => void;
  focusYear: (year: number) => void;
  resetAll: (firstYear: number, lastYear: number) => void;
};

export const useLinkedStore = create<LinkedState>((set) => ({
  activeView: "map",
  selectedProvince: null,
  selectedProduct: "",
  timeMode: "all",
  timeLayer: "event",
  animationKind: "annual",
  selectedYear: 2026,
  rangeStart: 2017,
  rangeEnd: 2026,
  isPlaying: false,
  setActiveView: (activeView) => set({ activeView }),
  setSelectedProvince: (selectedProvince) => set({ selectedProvince }),
  setSelectedProduct: (selectedProduct) => set({ selectedProduct }),
  setTimeMode: (timeMode) => set({ timeMode, isPlaying: false }),
  setTimeLayer: (timeLayer) => set({ timeLayer }),
  setAnimationKind: (animationKind) => set({ animationKind }),
  setSelectedYear: (selectedYear) => set({ selectedYear }),
  setRangeStart: (rangeStart) => set({ rangeStart }),
  setRangeEnd: (rangeEnd) => set({ rangeEnd }),
  setIsPlaying: (isPlaying) => set({ isPlaying }),
  focusYear: (year) => set({ timeMode: "single", selectedYear: year, isPlaying: false }),
  resetAll: (firstYear, lastYear) => set({
    selectedProvince: null, selectedProduct: "", timeMode: "all", timeLayer: "event",
    animationKind: "annual", selectedYear: lastYear, rangeStart: firstYear,
    rangeEnd: lastYear, isPlaying: false,
  }),
}));
