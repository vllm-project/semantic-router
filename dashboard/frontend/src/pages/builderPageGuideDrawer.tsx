import React, { useId } from "react";
import { createPortal } from "react-dom";

import DslGuide from "@/components/DslGuide";
import useAccessibleDialog from "@/hooks/useAccessibleDialog";

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
  const dialogId = useId();
  const titleId = `${dialogId}-title`;
  const dialogRef = useAccessibleDialog<HTMLDivElement>({
    isOpen: open,
    onClose,
    dismissible: !isDragging,
  });

  if (!open) return null;

  return createPortal(
    <div
      className={styles.guideDrawerOverlay}
      role="presentation"
      onMouseDown={() => {
        if (!isDragging) onClose();
      }}
    >
      <div
        ref={dialogRef}
        className={styles.guideDrawer}
        style={{ width }}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        tabIndex={-1}
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div
          className={styles.guideDrawerResizeHandle}
          onMouseDown={onDragStart}
        >
          <div className={styles.guideDrawerResizeLine} />
        </div>
        <div className={styles.guideDrawerHeader}>
          <h2 id={titleId} className={styles.guideDrawerTitle}>
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
          </h2>
          <button
            type="button"
            className={styles.guideDrawerClose}
            onClick={onClose}
            title="Close Guide"
            aria-label="Close DSL language guide"
            data-dialog-initial-focus
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
