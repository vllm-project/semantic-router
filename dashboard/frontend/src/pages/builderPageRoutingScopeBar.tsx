import React from "react";

import styles from "./BuilderPage.module.css";
import type { BuilderRoutingScope } from "./builderPageRoutingScopeSupport";

interface BuilderRoutingScopeBarProps {
  scopes: BuilderRoutingScope[];
  activeScopeId: string;
  onChange: (scopeId: string) => void;
}

export const BuilderRoutingScopeBar: React.FC<BuilderRoutingScopeBarProps> = ({
  scopes,
  activeScopeId,
  onChange,
}) => {
  if (scopes.length <= 1) return null;

  return (
    <div className={styles.routingScopeBar}>
      <span className={styles.routingScopeLabel}>Routing scope</span>
      <div className={styles.routingScopeOptions}>
        {scopes.map((scope) => (
          <button
            key={scope.id}
            type="button"
            className={
              scope.id === activeScopeId
                ? styles.routingScopeOptionActive
                : styles.routingScopeOption
            }
            title={scope.modelNames.length > 0 ? scope.modelNames.join(", ") : scope.label}
            onClick={() => onChange(scope.id)}
          >
            <span>{scope.label}</span>
            {scope.modelNames.length > 0 && (
              <span className={styles.routingScopeEntrypointCount}>{scope.modelNames.length}</span>
            )}
          </button>
        ))}
      </div>
    </div>
  );
};
