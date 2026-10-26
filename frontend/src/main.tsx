import { lazy, StrictMode, Suspense, type ReactNode } from "react";
import { createRoot } from "react-dom/client";
import { createBrowserRouter, RouterProvider } from "react-router-dom";
import { Layout } from "./components/Layout";
import "./index.css";
import { Spinner } from "./components/ui";
import { Dashboard } from "./pages/Dashboard";
import { NewAnalysis } from "./pages/NewAnalysis";
import { NotFound, RouteError } from "./pages/NotFound";

// Heavier pages (charts, video overlays) are code-split.
const AnalysisPage = lazy(() => import("./pages/Analysis").then((m) => ({ default: m.AnalysisPage })));
const ComparePage = lazy(() => import("./pages/Compare").then((m) => ({ default: m.ComparePage })));
const ExportPage = lazy(() => import("./pages/ExportPage").then((m) => ({ default: m.ExportPage })));
const SessionSetup = lazy(() => import("./pages/SessionSetup").then((m) => ({ default: m.SessionSetup })));

const page = (node: ReactNode) => <Suspense fallback={<Spinner />}>{node}</Suspense>;

const router = createBrowserRouter([
  {
    element: <Layout />,
    errorElement: <RouteError />,
    children: [
      { path: "/", element: <Dashboard /> },
      { path: "/new", element: <NewAnalysis /> },
      { path: "/sessions/:id/setup", element: page(<SessionSetup />) },
      { path: "/sessions/:id", element: page(<AnalysisPage />) },
      { path: "/sessions/:id/export", element: page(<ExportPage />) },
      { path: "/compare", element: page(<ComparePage />) },
      { path: "*", element: <NotFound /> },
    ],
  },
]);

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <RouterProvider router={router} />
  </StrictMode>,
);
