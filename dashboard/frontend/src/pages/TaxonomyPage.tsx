import { useState } from 'react'
import { useReadonly } from '../contexts/ReadonlyContext'
import ConfigPageManagerLayout from './ConfigPageManagerLayout'
import ConfigPageTaxonomyClassifiers from './ConfigPageTaxonomyClassifiers'
import EditModal, { type EditFormData, type FieldConfig } from '../components/EditModal'
import type { OpenEditModal } from './configPageRouterSectionSupport'

export type KnowledgeBaseView = 'bases' | 'groups' | 'labels'

interface TaxonomyPageProps {
  activeView: KnowledgeBaseView
}

const VIEW_META: Record<KnowledgeBaseView, { title: string; description: string }> = {
  bases: {
    title: 'Knowledge Bases',
    description: 'Manage the active knowledge base catalog.',
  },
  groups: {
    title: 'Knowledge Groups',
    description: 'Review paged group bindings for one base at a time.',
  },
  labels: {
    title: 'Knowledge Labels',
    description: 'Review label definitions and thresholds with a paged view.',
  },
}

export default function TaxonomyPage({ activeView }: TaxonomyPageProps) {
  const { isReadonly } = useReadonly()
  const [editModalOpen, setEditModalOpen] = useState(false)
  const [editModalTitle, setEditModalTitle] = useState('')
  const [editModalData, setEditModalData] = useState<EditFormData | null>(null)
  const [editModalFields, setEditModalFields] = useState<FieldConfig[]>([])
  const [editModalMode, setEditModalMode] = useState<'edit' | 'add'>('edit')
  const [editModalCallback, setEditModalCallback] = useState<
    ((data: EditFormData) => Promise<void>) | null
  >(null)

  const openEditModal: OpenEditModal = (title, data, fields, callback, mode = 'edit') => {
    setEditModalTitle(title)
    setEditModalData(data as EditFormData)
    setEditModalFields(fields as FieldConfig[])
    setEditModalMode(mode)
    setEditModalCallback(() => async (rawData: EditFormData) => callback(rawData as never))
    setEditModalOpen(true)
  }

  const closeEditModal = () => {
    setEditModalOpen(false)
    setEditModalData(null)
    setEditModalFields([])
    setEditModalCallback(null)
  }

  const meta = VIEW_META[activeView]

  return (
    <>
      <ConfigPageManagerLayout
        eyebrow="Knowledge"
        title={meta.title}
        description={meta.description}
      >
        <ConfigPageTaxonomyClassifiers
          isReadonly={isReadonly}
          openEditModal={openEditModal}
          activeView={activeView}
        />
      </ConfigPageManagerLayout>

      <EditModal
        isOpen={editModalOpen}
        onClose={closeEditModal}
        onSave={editModalCallback || (async () => {})}
        title={editModalTitle}
        data={editModalData}
        fields={editModalFields}
        mode={editModalMode}
      />
    </>
  )
}
