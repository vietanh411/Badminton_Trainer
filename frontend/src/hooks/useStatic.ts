import { useEffect, useState } from "react";
import { api } from "../services/api";
import type { CourtGeometry, SystemInfo } from "../types/api";

let courtPromise: Promise<CourtGeometry> | null = null;
let systemPromise: Promise<SystemInfo> | null = null;

/** Court geometry never changes during a session: fetch once and share. */
export function useCourt(): CourtGeometry | null {
  const [court, setCourt] = useState<CourtGeometry | null>(null);
  useEffect(() => {
    courtPromise ??= api.court();
    courtPromise.then(setCourt, () => {
      courtPromise = null;
    });
  }, []);
  return court;
}

export function useSystem(): SystemInfo | null {
  const [sys, setSys] = useState<SystemInfo | null>(null);
  useEffect(() => {
    systemPromise ??= api.system();
    systemPromise.then(setSys, () => {
      systemPromise = null;
    });
  }, []);
  return sys;
}
