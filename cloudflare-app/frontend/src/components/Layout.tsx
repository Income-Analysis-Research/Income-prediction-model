import { useState, useEffect } from "react";
import { Outlet, NavLink, useLocation } from "react-router-dom";
import {
  HomeIcon,
  BeakerIcon,
  MapIcon,
  ChartBarIcon,
  MagnifyingGlassCircleIcon,
  ClipboardDocumentListIcon,
  Bars3Icon,
  XMarkIcon,
  SignalIcon,
} from "@heroicons/react/24/outline";
import { api } from "../lib/api";

const NAV = [
  { to: "/dashboard",  label: "Overview",         Icon: HomeIcon },
  { to: "/model",      label: "Model Parameters", Icon: BeakerIcon },
  { to: "/geographic", label: "Geographic",        Icon: MapIcon },
  { to: "/temporal",   label: "Temporal",          Icon: ChartBarIcon },
  { to: "/predictor",  label: "Income Predictor",  Icon: MagnifyingGlassCircleIcon },
  { to: "/audit",      label: "Audit Log",         Icon: ClipboardDocumentListIcon },
];

export default function Layout() {
  const [open,   setOpen]   = useState(false);
  const [health, setHealth] = useState<"ok" | "error" | "loading">("loading");
  const location = useLocation();

  useEffect(() => {
    setOpen(false);
  }, [location]);

  useEffect(() => {
    api.health()
      .then(() => setHealth("ok"))
      .catch(() => setHealth("error"));
  }, []);

  return (
    <div className="flex h-screen overflow-hidden">
      {/* ── Mobile overlay ── */}
      {open && (
        <div
          className="fixed inset-0 bg-black/60 z-30 lg:hidden"
          onClick={() => setOpen(false)}
        />
      )}

      {/* ── Sidebar ── */}
      <aside
        className={[
          "fixed inset-y-0 left-0 z-40 w-64 flex flex-col",
          "bg-navy-900 border-r border-navy-600/40",
          "transform transition-transform duration-300 ease-in-out lg:translate-x-0 lg:static lg:inset-0",
          open ? "translate-x-0" : "-translate-x-full",
        ].join(" ")}
      >
        {/* Logo */}
        <div className="flex items-center gap-3 px-5 py-5 border-b border-navy-600/40">
          <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center text-white font-bold text-sm flex-shrink-0">
            IQ
          </div>
          <div>
            <div className="text-sm font-bold text-slate-100 tracking-tight">IncomeIQ</div>
            <div className="text-[10px] font-medium text-slate-500 tracking-widest uppercase">
              Regional Intelligence
            </div>
          </div>
        </div>

        {/* Nav */}
        <nav className="flex-1 overflow-y-auto py-4 px-3 space-y-1">
          <p className="label px-2 mb-3">Navigation</p>
          {NAV.map(({ to, label, Icon }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                isActive ? "nav-item-active" : "nav-item"
              }
            >
              <Icon className="w-4 h-4 flex-shrink-0" />
              {label}
            </NavLink>
          ))}
        </nav>

        {/* Footer */}
        <div className="border-t border-navy-600/40 px-4 py-4">
          <div className="flex items-center gap-2 mb-2">
            <SignalIcon className="w-4 h-4 text-slate-500" />
            <span className="text-xs text-slate-500">API Status</span>
            <span
              className={[
                "ml-auto w-2 h-2 rounded-full",
                health === "ok"      ? "bg-emerald-400 shadow-[0_0_6px_rgba(52,211,153,0.6)]" :
                health === "loading" ? "bg-amber-400 animate-pulse" :
                                       "bg-red-400",
              ].join(" ")}
            />
          </div>
          <p className="text-[10px] text-slate-600 leading-relaxed">
            Hierarchical Bayesian Panel Regression<br />
            IRS SOI ZIP Data 2011–2022
          </p>
        </div>
      </aside>

      {/* ── Main ── */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        {/* Top bar */}
        <header className="flex-shrink-0 h-14 flex items-center justify-between px-4 lg:px-6
                           bg-navy-900/80 backdrop-blur-sm border-b border-navy-600/40 z-20">
          <div className="flex items-center gap-3">
            <button
              onClick={() => setOpen(!open)}
              className="lg:hidden p-2 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-navy-700/60 transition-colors"
            >
              {open ? <XMarkIcon className="w-5 h-5" /> : <Bars3Icon className="w-5 h-5" />}
            </button>
            <div>
              <span className="text-sm font-semibold text-slate-200 hidden sm:block">
                {NAV.find((n) => location.pathname.startsWith(n.to))?.label ?? "IncomeIQ"}
              </span>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <span className="badge badge-blue hidden sm:inline-flex">Bayesian Model v1.0</span>
            <span className="badge badge-green">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 mr-1" />
              Live
            </span>
          </div>
        </header>

        {/* Page content */}
        <main className="flex-1 overflow-y-auto">
          <div className="p-4 lg:p-6 max-w-7xl mx-auto animate-fade-in">
            <Outlet />
          </div>
        </main>
      </div>

      {/* ── Mobile bottom tab bar ── */}
      <div className="lg:hidden fixed bottom-0 inset-x-0 z-30
                      bg-navy-900/95 backdrop-blur-md border-t border-navy-600/40">
        <div className="flex">
          {NAV.slice(0, 5).map(({ to, label, Icon }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                [
                  "flex-1 flex flex-col items-center py-2 px-1 text-[10px] font-medium transition-colors",
                  isActive
                    ? "text-blue-400"
                    : "text-slate-500 hover:text-slate-300",
                ].join(" ")
              }
            >
              <Icon className="w-5 h-5 mb-1" />
              <span className="truncate w-full text-center">{label.split(" ")[0]}</span>
            </NavLink>
          ))}
        </div>
      </div>
    </div>
  );
}
