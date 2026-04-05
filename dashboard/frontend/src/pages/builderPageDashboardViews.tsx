import React from "react";

import type { BoolExprNode, EditorMode } from "@/types/dsl";
import { useDSLStore } from "@/stores/dslStore";

import styles from "./BuilderPage.module.css";
import {
  ModelIcon,
  PluginIcon,
  RouteIcon,
  SignalIcon,
} from "./builderPageFormPrimitives";
import type { EntityKind, Selection } from "./builderPageTypes";

interface SidebarSectionProps {
  title: string;
  count: number;
  open: boolean;
  onToggle: () => void;
  onAdd?: () => void;
  children: React.ReactNode;
}

const SidebarSection: React.FC<SidebarSectionProps> = ({
  title,
  count,
  open,
  onToggle,
  onAdd,
  children,
}) => (
  <div className={styles.sidebarSection}>
    <div className={styles.sidebarSectionHeader} onClick={onToggle}>
      <span className={styles.sidebarSectionTitle}>
        {title}
        <span className={styles.sidebarCount}>{count}</span>
      </span>
      <span style={{ display: "flex", alignItems: "center", gap: "0.25rem" }}>
        {onAdd && (
          <button
            className={styles.sidebarAddBtn}
            onClick={(e) => {
              e.stopPropagation();
              onAdd();
            }}
            title={`Add ${title.slice(0, -1)}`}
            style={{ width: "auto", padding: "0.125rem 0.25rem" }}
          >
            +
          </button>
        )}
        <svg
          className={`${styles.sidebarSectionChevron} ${open ? styles.sidebarSectionChevronOpen : ""}`}
          viewBox="0 0 16 16"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
        >
          <path d="M6 4l4 4-4 4" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </span>
    </div>
    {open && <ul className={styles.sidebarList}>{children}</ul>}
  </div>
);

// ===================================================================
// Dashboard View (no entity selected)
// ===================================================================

interface DashboardViewProps {
  ast: ReturnType<typeof useDSLStore.getState>["ast"];
  modelCount: number;
  signalCount: number;
  routeCount: number;
  pluginCount: number;
  isValid: boolean;
  errorCount: number;
  onSelect: (sel: Selection) => void;
  onAddEntity: (kind: EntityKind) => void;
  onModeSwitch: (mode: EditorMode) => void;
}

/** Serialize a BoolExprNode into a short readable string */
function boolExprToText(node: BoolExprNode | null, maxLen = 60): string {
  if (!node) return "(always)";
  const serialize = (n: BoolExprNode): string => {
    switch (n.type) {
      case "signal_ref":
        return `${n.signalType}("${n.signalName}")`;
      case "not":
        return `NOT ${serialize(n.expr)}`;
      case "and": {
        const l = serialize(n.left),
          r = serialize(n.right);
        return `${l} AND ${r}`;
      }
      case "or": {
        const l = serialize(n.left),
          r = serialize(n.right);
        return `(${l} OR ${r})`;
      }
    }
  };
  const text = serialize(node);
  return text.length > maxLen ? text.slice(0, maxLen - 3) + "..." : text;
}

const DashboardView: React.FC<DashboardViewProps> = ({
  ast,
  modelCount,
  signalCount,
  routeCount,
  pluginCount,
  isValid,
  errorCount,
  onSelect,
  onAddEntity,
  onModeSwitch,
}) => {
  const routes = ast?.routes ?? [];
  const defaultRoute = routes.find((r) => !r.when);
  const conditionalRoutes = routes
    .filter((r) => !!r.when)
    .sort((a, b) => b.priority - a.priority);

  return (
    <div className={styles.dashboard}>
      {/* Title */}
      <div className={styles.dashboardHeader}>
        <div className={styles.dashboardTitle}>
          <svg
            width="20"
            height="20"
            viewBox="0 0 16 16"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
          >
            <rect x="2" y="2" width="5" height="5" rx="1" />
            <rect x="9" y="2" width="5" height="5" rx="1" />
            <rect x="2" y="9" width="5" height="5" rx="1" />
            <rect x="9" y="9" width="5" height="5" rx="1" />
          </svg>
          Semantic Router Config Builder
        </div>
        <div
          className={`${styles.dashboardBadge} ${isValid ? styles.dashboardBadgeOk : styles.dashboardBadgeErr}`}
        >
          {isValid
            ? "✓ Valid"
            : `${errorCount} error${errorCount !== 1 ? "s" : ""}`}
        </div>
      </div>

      {/* Status Cards */}
      <div className={styles.statsGrid}>
        {[
          {
            label: "Models",
            count: modelCount,
            kind: "model" as const,
            icon: <ModelIcon className={styles.statIcon} />,
          },
          {
            label: "Signals",
            count: signalCount,
            kind: "signal" as const,
            icon: <SignalIcon className={styles.statIcon} />,
          },
          {
            label: "Routes",
            count: routeCount,
            kind: "route" as const,
            icon: <RouteIcon className={styles.statIcon} />,
          },
          {
            label: "Plugins",
            count: pluginCount,
            kind: "plugin" as const,
            icon: <PluginIcon className={styles.statIcon} />,
          },
        ].map((card) => (
          <div
            key={card.label}
            className={styles.statCard}
            onClick={() =>
              card.count > 0 &&
              onSelect({
                kind: card.kind,
                name: "__list__",
              })
            }
          >
            {card.icon}
            <span className={styles.statValue}>{card.count}</span>
            <span className={styles.statLabel}>{card.label}</span>
            <span
              className={`${styles.statBadge} ${card.count > 0 ? styles.statBadgeOk : styles.statBadgeEmpty}`}
            >
              {card.count > 0 ? "✓ valid" : "empty"}
            </span>
          </div>
        ))}
      </div>

      {/* Quick Actions */}
      <div className={styles.dashSection}>
        <div className={styles.dashSectionTitle}>Quick Actions</div>
        <div className={styles.quickActions}>
          <button
            className={styles.quickActionBtn}
            onClick={() => onAddEntity("model")}
          >
            <span className={styles.quickActionIcon}>+</span> New Model
          </button>
          <button
            className={styles.quickActionBtn}
            onClick={() => onAddEntity("signal")}
          >
            <span className={styles.quickActionIcon}>+</span> New Signal
          </button>
          <button
            className={styles.quickActionBtn}
            onClick={() => onAddEntity("route")}
          >
            <span className={styles.quickActionIcon}>+</span> New Route
          </button>
          <button
            className={styles.quickActionBtn}
            onClick={() => onAddEntity("plugin")}
          >
            <span className={styles.quickActionIcon}>+</span> New Plugin
          </button>
        </div>
      </div>

      {/* Route Map (visual flow) */}
      {routes.length > 0 && (
        <div className={styles.dashSection}>
          <div className={styles.dashSectionTitle}>Route Map</div>
          <div className={styles.routeMap}>
            <div className={styles.routeMapEntry}>
              <span className={styles.routeMapEntryLabel}>User Query</span>
            </div>
            <div className={styles.routeMapFlow}>
              {conditionalRoutes.map((route) => (
                <div
                  key={route.name}
                  className={styles.routeMapBranch}
                  onClick={() => onSelect({ kind: "route", name: route.name })}
                >
                  <div className={styles.routeMapCondition}>
                    <span className={styles.routeMapCondIcon}>├─</span>
                    <code className={styles.routeMapCondText}>
                      {boolExprToText(route.when)}
                    </code>
                  </div>
                  <div className={styles.routeMapTarget}>
                    <span className={styles.routeMapTargetArrow}>└→</span>
                    <span className={styles.routeMapRouteName}>
                      &quot;{route.name}&quot;
                    </span>
                    <span className={styles.routeMapTargetArrow}>→</span>
                    <span className={styles.routeMapModel}>
                      {route.models.length > 0
                        ? route.models.map((m) => m.model).join(", ")
                        : "(no model)"}
                    </span>
                  </div>
                </div>
              ))}
              {defaultRoute && (
                <div
                  className={styles.routeMapBranch}
                  onClick={() =>
                    onSelect({ kind: "route", name: defaultRoute.name })
                  }
                >
                  <div className={styles.routeMapCondition}>
                    <span className={styles.routeMapCondIcon}>└─</span>
                    <code className={styles.routeMapCondText}>
                      (no match / default)
                    </code>
                  </div>
                  <div className={styles.routeMapTarget}>
                    <span className={styles.routeMapTargetArrow}>└→</span>
                    <span className={styles.routeMapRouteName}>
                      &quot;{defaultRoute.name}&quot;
                    </span>
                    <span className={styles.routeMapTargetArrow}>→</span>
                    <span className={styles.routeMapModel}>
                      {defaultRoute.models.length > 0
                        ? defaultRoute.models.map((m) => m.model).join(", ")
                        : "(no model)"}
                    </span>
                  </div>
                </div>
              )}
              {routes.length === 0 && (
                <div className={styles.routeMapEmpty}>
                  No routes defined yet
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Mode Switcher */}
      <div className={styles.dashSection}>
        <div className={styles.dashSectionTitle}>Editor Mode</div>
        <div className={styles.dashModes}>
          <button
            className={styles.dashModeBtn}
            onClick={() => onModeSwitch("visual")}
          >
            <span className={styles.dashModeBtnIcon}>📐</span>
            <span className={styles.dashModeBtnLabel}>Visual</span>
            <span className={styles.dashModeBtnDesc}>Drag & drop builder</span>
          </button>
          <button
            className={styles.dashModeBtn}
            onClick={() => onModeSwitch("dsl")}
          >
            <span className={styles.dashModeBtnIcon}>📝</span>
            <span className={styles.dashModeBtnLabel}>DSL</span>
            <span className={styles.dashModeBtnDesc}>Code editor</span>
          </button>
          <button
            className={styles.dashModeBtn}
            onClick={() => onModeSwitch("nl")}
          >
            <span className={styles.dashModeBtnIcon}>🤖</span>
            <span className={styles.dashModeBtnLabel}>Natural Language</span>
            <span className={styles.dashModeBtnDesc}>
              Prompt-driven DSL draft generation
            </span>
          </button>
        </div>
      </div>
    </div>
  );
};

// ===================================================================
// Entity List View (shows all entities of a kind as a card grid)
// ===================================================================

interface EntityListViewProps {
  kind: EntityKind;
  ast: ReturnType<typeof useDSLStore.getState>["ast"];
  onSelect: (sel: Selection) => void;
  onBack: () => void;
  onAddEntity: (kind: EntityKind) => void;
}

const EntityListView: React.FC<EntityListViewProps> = ({
  kind,
  ast,
  onSelect,
  onBack,
  onAddEntity,
}) => {
  const META: Record<
    string,
    { title: string; icon: React.FC<{ className?: string }>; color: string }
  > = {
    model: { title: "Models", icon: ModelIcon, color: "rgb(251, 191, 36)" },
    signal: { title: "Signals", icon: SignalIcon, color: "rgb(118, 185, 0)" },
    route: { title: "Routes", icon: RouteIcon, color: "rgb(96, 165, 250)" },
    plugin: { title: "Plugins", icon: PluginIcon, color: "rgb(168, 130, 255)" },
  };
  const meta = META[kind];
  if (!meta) return null;
  const Icon = meta.icon;

  const items: { name: string; type: string; desc?: string }[] = (() => {
    switch (kind) {
      case "model":
        return (ast?.models ?? []).map((model) => ({
          name: model.name,
          type: "catalog",
          desc:
            Object.keys(model.fields).length > 0
              ? `${Object.keys(model.fields).length} field(s)`
              : undefined,
        }));
      case "signal":
        return (ast?.signals ?? []).map((s) => ({
          name: s.name,
          type: s.signalType,
          desc:
            Object.keys(s.fields).length > 0
              ? `${Object.keys(s.fields).length} field(s)`
              : undefined,
        }));
      case "route":
        return (ast?.routes ?? []).map((r) => ({
          name: r.name,
          type: r.when ? `P${r.priority}` : "default",
          desc:
            r.models.length > 0
              ? r.models.map((m) => m.model).join(", ")
              : undefined,
        }));
      case "plugin":
        return (ast?.plugins ?? []).map((p) => ({
          name: p.name,
          type: p.pluginType,
          desc:
            Object.keys(p.fields).length > 0
              ? `${Object.keys(p.fields).length} field(s)`
              : undefined,
        }));
      default:
        return [];
    }
  })();

  return (
    <div className={styles.entityListPanel}>
      <div className={styles.entityListHeader}>
        <button
          className={styles.backBtn}
          onClick={onBack}
          title="Back to Dashboard"
        >
          <svg
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <polyline points="15 18 9 12 15 6" />
          </svg>
        </button>
        <Icon className={styles.statIcon} />
        <span className={styles.entityListTitle}>{meta.title}</span>
        <span className={styles.entityListCount}>{items.length}</span>
        <div style={{ marginLeft: "auto" }}>
          <button
            className={styles.quickActionBtn}
            onClick={() => onAddEntity(kind)}
            style={{ padding: "0.5rem 1rem", fontSize: "0.8125rem" }}
          >
            <span
              className={styles.quickActionIcon}
              style={{ width: 24, height: 24, fontSize: "0.875rem" }}
            >
              +
            </span>
            New {meta.title.replace(/s$/, "")}
          </button>
        </div>
      </div>
      <div className={styles.entityListGrid}>
        {items.map((item) => (
          <div
            key={item.name}
            className={styles.entityListCard}
            onClick={() => onSelect({ kind, name: item.name })}
            style={{ "--entity-accent": meta.color } as React.CSSProperties}
          >
            <div className={styles.entityListCardHeader}>
              <Icon className={styles.entityListCardIcon} />
              <span className={styles.entityListCardName}>{item.name}</span>
            </div>
            <span className={styles.entityListCardType}>{item.type}</span>
            {item.desc && (
              <span className={styles.entityListCardDesc}>{item.desc}</span>
            )}
            <div className={styles.entityListCardArrow}>
              <svg
                width="14"
                height="14"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <polyline points="9 6 15 12 9 18" />
              </svg>
            </div>
          </div>
        ))}
      </div>
      {items.length === 0 && (
        <div className={styles.emptyState}>
          <div className={styles.emptyIcon}>
            <Icon className={styles.statIcon} />
          </div>
          <div>No {meta.title.toLowerCase()} defined yet</div>
          <div
            style={{
              fontSize: "var(--text-xs)",
              color: "var(--color-text-muted)",
            }}
          >
            Click the button above to create one
          </div>
        </div>
      )}
    </div>
  );
};

export { DashboardView, EntityListView, SidebarSection };

// ===================================================================
// Entity Detail View (editable for Phase 2)
// ===================================================================
