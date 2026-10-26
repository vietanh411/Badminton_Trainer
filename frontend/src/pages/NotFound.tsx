import { isRouteErrorResponse, Link, useRouteError } from "react-router-dom";
import { EmptyState, ErrorBox } from "../components/ui";

export function NotFound() {
  return (
    <EmptyState title="Page not found">
      <Link className="text-accent hover:underline" to="/">
        Back to the dashboard
      </Link>
    </EmptyState>
  );
}

/** Shown instead of a blank page when a route throws while rendering. */
export function RouteError() {
  const err = useRouteError();
  const msg = isRouteErrorResponse(err) ? `${err.status} ${err.statusText}` : err instanceof Error ? err.message : String(err);
  return (
    <div className="mx-auto max-w-2xl space-y-4 p-8">
      <h1 className="text-xl font-semibold text-white">Something went wrong on this page</h1>
      <ErrorBox error={msg} />
      <Link className="text-accent hover:underline" to="/">
        Back to the dashboard
      </Link>
    </div>
  );
}
