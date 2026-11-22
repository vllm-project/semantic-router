import React from 'react'

import {
  DetailSheet,
  DetailSheetBody,
  DetailSheetFooter,
  DetailSheetHeader,
  useRetainedValue,
} from '../ui'
import {
  ViewPanelActions,
  ViewSections,
  type ViewField,
  type ViewPanelAction,
  type ViewSection,
} from './ViewPanel'
import { transitionFromViewToEdit } from './viewModalSupport'

interface ViewModalProps {
  isOpen: boolean
  onClose: () => void
  onEdit?: () => void
  title: string
  sections: ViewSection[]
  actions?: ViewPanelAction[]
  /** Labels an extra footer button that closes the sheet; the header always has one. */
  closeLabel?: string
}

const ViewModal: React.FC<ViewModalProps> = ({
  isOpen,
  onClose,
  onEdit,
  title,
  sections,
  actions = [],
  closeLabel,
}) => {
  // Callers clear title and sections on close; keep them while the sheet animates out.
  const shownTitle = useRetainedValue(isOpen ? title : null) ?? title
  const shownSections = useRetainedValue(isOpen ? sections : null) ?? sections
  const handleEdit = onEdit ? () => transitionFromViewToEdit(onClose, onEdit) : undefined
  const hasFooter = actions.length > 0 || Boolean(handleEdit) || Boolean(closeLabel)

  return (
    <DetailSheet
      open={isOpen}
      onOpenChange={(open) => {
        if (!open) onClose()
      }}
    >
      <DetailSheetHeader title={shownTitle} />
      <DetailSheetBody className="px-5 py-5">
        <ViewSections sections={shownSections} />
      </DetailSheetBody>
      {hasFooter ? (
        <DetailSheetFooter className="flex-wrap justify-end">
          <ViewPanelActions
            actions={actions}
            onEdit={handleEdit}
            onClose={onClose}
            closeLabel={closeLabel}
          />
        </DetailSheetFooter>
      ) : null}
    </DetailSheet>
  )
}

export type { ViewField, ViewPanelAction, ViewSection }
export default ViewModal
