import { useEffect, useRef, useState } from "react";
import { api, jobEventsUrl } from "../services/api";
import { TERMINAL, type Job } from "../types/api";

/**
 * Follow a job's progress with Server-Sent Events, falling back to polling every second if the
 * stream fails (proxies that buffer SSE, older browsers...).
 */
export function useJob(jobId: string | null | undefined, onDone?: (job: Job) => void): Job | null {
  const [job, setJob] = useState<Job | null>(null);
  const done = useRef(onDone);
  done.current = onDone;

  useEffect(() => {
    if (!jobId) {
      setJob(null);
      return;
    }
    let closed = false;
    let es: EventSource | null = null;
    let timer: number | undefined;
    const handle = (j: Job) => {
      if (closed) return;
      setJob(j);
      if (TERMINAL.includes(j.status)) {
        closed = true;
        es?.close();
        window.clearInterval(timer);
        done.current?.(j);
      }
    };
    const poll = () => {
      timer = window.setInterval(() => {
        api.job(jobId).then(handle, () => undefined);
      }, 1000);
    };
    api.job(jobId).then(handle, () => undefined);
    if (typeof EventSource !== "undefined") {
      es = new EventSource(jobEventsUrl(jobId));
      es.onmessage = (ev) => {
        try {
          handle(JSON.parse(ev.data) as Job);
        } catch {
          /* ignore malformed */
        }
      };
      es.onerror = () => {
        es?.close();
        es = null;
        if (!closed && timer === undefined) poll();
      };
    } else poll();
    return () => {
      closed = true;
      es?.close();
      window.clearInterval(timer);
    };
  }, [jobId]);

  return job;
}
