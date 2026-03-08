import React from "react";
import { createPortal } from "react-dom";

import DslGuide from "@/components/DslGuide";

import styles from "./BuilderPage.module.css";

interface BuilderGuideDrawerProps {
  open: boolean;
  width: number;
  isDragging: boolean;
  onClose: () => void;
  onDragStart: (event: React.MouseEvent) => void;
  onInsertSnippet: (snippet: string) => void;
}

const BuilderGuideDrawer: React.FC<BuilderGuideDrawerProps> = ({
  open,
  width,
  isDragging,
  onClose,
  onDragStart,
  onInsertSnippet,
}) => {
  if (!open) return null;

  return createPortal(
    <div
      className={styles.guideDrawerOverlay}
      onClick={() => {
        if (!isDragging) onClose();
      }}
    >
      <div
        className={styles.guideDrawer}
        style={{ width }}
        onClick={(event) => event.stopPropagation()}
      >
        <div
          className={styles.guideDrawerResizeHandle}
          onMouseDown={onDragStart}
        >
          <div className={styles.guideDrawerResizeLine} />
        </div>
        <div className={styles.guideDrawerHeader}>
          <span className={styles.guideDrawerTitle}>
            <svg
              width="14"
              height="14"
              viewBox="0 0 16 16"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
            >
              <path
                d="M2 2h9a2 2 0 012 2v10l-3-2H2V2z"
                strokeLinejoin="round"
              />
              <path d="M5 6h5M5 9h3" strokeLinecap="round" />
            </svg>
            DSL Language Guide
          </span>
          <button
            className={styles.guideDrawerClose}
            onClick={onClose}
            title="Close Guide"
          >
            <svg
              width="14"
              height="14"
              viewBox="0 0 16 16"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <path d="M4 4l8 8M12 4l-8 8" strokeLinecap="round" />
            </svg>
          </button>
        </div>
        <div className={styles.guideDrawerBody}>
          <DslGuide onInsertSnippet={onInsertSnippet} />
        </div>
      </div>
    </div>,
    document.body,
  );
};

export { BuilderGuideDrawer };
