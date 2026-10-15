import { useLinkedStore } from "../store/useLinkedStore";

const items = [
  ["map", "⌖", "Ana Harita"],
  ["stc", "◫", "STC"],
  ["climate", "◉", "İklim"],
  ["interventions", "◇", "Müdahaleler"],
  ["documents", "▤", "Belgeler"],
  ["questions", "?", "Sorular"],
  ["methodology", "⚙", "Metodoloji"],
] as const;
export default function Sidebar() {
  const activeView = useLinkedStore((s) => s.activeView);
  const setActiveView = useLinkedStore((s) => s.setActiveView);

  return (
    <aside className="sidebar">
      <div className="brand-mark"><div className="bug-mark">🐞</div></div>
      <nav className="sidebar-nav">
        {items.map(([key, icon, label]) => (
          <button
            key={key}
            className={activeView === key ? "sidebar-item active" : "sidebar-item"}
            onClick={() => setActiveView(key)}
            title={label}
          >
            <span className="sidebar-icon">{icon}</span>
            <span className="sidebar-label">{label}</span>
          </button>
        ))}
      </nav>
      <div className="sidebar-footer"><div className="status-dot" /><span>Linked</span></div>
    </aside>
  );
}
