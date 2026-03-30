import { Navigate, Route, Routes } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { getMetricsMeta } from "./lib/api";
import { AppShell } from "./components/AppShell";
import { AnalyticalPage } from "./pages/AnalyticalPage";
import { ClimateHealthRelationshipsPage } from "./pages/ClimateHealthRelationshipsPage";
import { HowToUsePage } from "./pages/HowToUsePage";
import { MethodsPage } from "./pages/MethodsPage";
import { OutcomeMatrixPage } from "./pages/OutcomeMatrixPage";

function App() {
  const { data, error, isLoading } = useQuery({ queryKey: ["metrics-meta"], queryFn: getMetricsMeta });

  if (isLoading) {
    return <div className="p-6 text-sm text-slate-500">Loading application metadata...</div>;
  }

  if (error || !data) {
    return (
      <div className="p-6 text-sm text-caution">
        Unable to load application metadata. Check that the dashboard API is running on port 8000.
      </div>
    );
  }

  return (
    <Routes>
      <Route path="/" element={<AppShell pages={data.pages} />}>
        <Route index element={<Navigate to="/overview" replace />} />
        <Route path="overview" element={<AnalyticalPage pageId="overview" />} />
        <Route path="perinatal" element={<AnalyticalPage pageId="perinatal" />} />
        <Route path="child-mortality" element={<AnalyticalPage pageId="child_mortality" />} />
        <Route path="maternal" element={<AnalyticalPage pageId="maternal" />} />
        <Route path="climate" element={<AnalyticalPage pageId="climate" />} />
        <Route path="context" element={<AnalyticalPage pageId="context" />} />
        <Route path="relationships" element={<ClimateHealthRelationshipsPage />} />
        <Route path="how-to-use" element={<HowToUsePage />} />
        <Route path="outcome-matrix" element={<OutcomeMatrixPage />} />
        <Route path="methods" element={<MethodsPage />} />
      </Route>
    </Routes>
  );
}

export default App;
