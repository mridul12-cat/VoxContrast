import React from "react";

export type TabId = "overview" | "compare" | "flaws" | "dataset" | "system";

interface PrimaryNavigationProps {
  activeTab: TabId;
  onSelectTab: (tab: TabId) => void;
  flawCount?: number;
  sessionUploadCount?: number;
}

export const PrimaryNavigation: React.FC<PrimaryNavigationProps> = ({
  activeTab,
  onSelectTab,
  flawCount = 0,
  sessionUploadCount = 0,
}) => {
  const datasetBadge = sessionUploadCount > 0 ? `18 (+${sessionUploadCount})` : 18;

  const tabs: Array<{ id: TabId; label: string; badge?: string | number }> = [
    { id: "overview", label: "Overview" },
    { id: "compare", label: "Compare Audio" },
    { id: "flaws", label: "Flaw Analysis", badge: flawCount },
    { id: "dataset", label: "Dataset", badge: datasetBadge },
    { id: "system", label: "System", badge: "Specs & Eval" },
  ];

  const handleKeyDown = (e: React.KeyboardEvent<HTMLButtonElement>, index: number) => {
    let targetIndex: number | null = null;
    if (e.key === "ArrowRight" || e.key === "ArrowDown") {
      e.preventDefault();
      targetIndex = (index + 1) % tabs.length;
    } else if (e.key === "ArrowLeft" || e.key === "ArrowUp") {
      e.preventDefault();
      targetIndex = (index - 1 + tabs.length) % tabs.length;
    } else if (e.key === "Home") {
      e.preventDefault();
      targetIndex = 0;
    } else if (e.key === "End") {
      e.preventDefault();
      targetIndex = tabs.length - 1;
    }

    if (targetIndex !== null) {
      onSelectTab(tabs[targetIndex].id);
      const targetBtn = document.getElementById(`tab-btn-${tabs[targetIndex].id}`);
      if (targetBtn) targetBtn.focus();
    }
  };

  return (
    <nav className="border-t border-slate-800/80 bg-slate-900/60 px-4 sm:px-6 lg:px-8" aria-label="Primary Navigation">
      <div className="max-w-7xl mx-auto flex items-center space-x-1 sm:space-x-2 overflow-x-auto py-2" role="tablist">
        {tabs.map((tab, idx) => {
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              id={`tab-btn-${tab.id}`}
              role="tab"
              aria-selected={isActive}
              tabIndex={0}
              onClick={() => onSelectTab(tab.id)}
              onKeyDown={(e) => handleKeyDown(e, idx)}
              className={`px-3.5 py-2 rounded-xl text-xs font-semibold transition border flex items-center space-x-2 whitespace-nowrap focus:outline-none focus:ring-2 focus:ring-indigo-500 ${
                isActive
                  ? "bg-indigo-600/20 text-indigo-300 border-indigo-500/80 shadow-sm"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60 border-transparent"
              }`}
            >
              <span>{tab.label}</span>
              {tab.badge !== undefined && (
                <span
                  className={`px-1.5 py-0.2 rounded-full text-[10px] font-mono ${
                    tab.id === "flaws"
                      ? "bg-rose-500/20 text-rose-300 border border-rose-500/30"
                      : tab.id === "system"
                      ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30"
                      : "bg-indigo-500/20 text-indigo-300 border border-indigo-500/30"
                  }`}
                >
                  {tab.badge}
                </span>
              )}
            </button>
          );
        })}
      </div>
    </nav>
  );
};
