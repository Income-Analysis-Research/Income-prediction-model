import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import Layout from "./components/Layout";
import Dashboard   from "./pages/Dashboard";
import ModelParams from "./pages/ModelParams";
import Geographic  from "./pages/Geographic";
import Temporal    from "./pages/Temporal";
import Predictor   from "./pages/Predictor";
import AuditLog    from "./pages/AuditLog";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index             element={<Navigate to="/dashboard" replace />} />
          <Route path="/dashboard" element={<Dashboard   />} />
          <Route path="/model"     element={<ModelParams />} />
          <Route path="/geographic"element={<Geographic  />} />
          <Route path="/temporal"  element={<Temporal    />} />
          <Route path="/predictor" element={<Predictor   />} />
          <Route path="/audit"     element={<AuditLog    />} />
          <Route path="*"          element={<Navigate to="/dashboard" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
