import React from 'react'

import styles from './ViewModal.module.css'
import ViewPanel, { type ViewField, type ViewPanelAction, type ViewSection } from './ViewPanel'

interface ViewModalProps {
  isOpen: boolean
  onClose: () => void
  onEdit?: () => void
  title: string
  sections: ViewSection[]
  actions?: ViewPanelAction[]
  closeLabel?: string
}

const ViewModal: React.FC<ViewModalProps> = ({
  isOpen,
  onClose,
  onEdit,
  title,
  sections,
  actions,
  closeLabel,
}) => {
  if (!isOpen) return null

  return (
    <div className={styles.overlay} onClick={onClose}>
      <div onClick={(event) => event.stopPropagation()}>
        <ViewPanel
          title={title}
          sections={sections}
          onClose={onClose}
          onEdit={onEdit}
          actions={actions}
          closeLabel={closeLabel}
        />
      </div>
    </div>
  )
}

export type { ViewField, ViewPanelAction, ViewSection }
export default ViewModal
