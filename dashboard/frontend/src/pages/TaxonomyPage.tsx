import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
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
    description: 'Manage built-in and custom knowledge packages, keep the active catalog clean, and inspect only the settings that matter for routing.',
  },
  groups: {
    title: 'Knowledge Groups',
    description: 'Work one base at a time, paginate large group sets, and keep routing groups readable even when a knowledge base carries many labels.',
  },
  labels: {
    title: 'Knowledge Labels',
    description: 'Review label definitions, thresholds, and signal references with a paged view instead of exposing raw exemplar CRUD as its own surface.',
  },
}

export default function TaxonomyPage({ activeView }: TaxonomyPageProps) {
  const { isReadonly } = useReadonly()
  const navigate = useNavigate()
  const [editModalOpen, setEditModalOpen] = useState(false)
  const [editModalTitle, setEditModalTitle] = useState('')
  const [editModalData, setEditModalData] = useState<EditFormData | null>(null)
  const [editModalFields, setEditModalFields] = useState<FieldConfig[]>([])
  const [editModalMode, setEditModalMode] = useState<'edit' | 'add'>('edit')
  const [editModalCallback, setEditModalCallback] = useState<((data: EditFormData) => Promise<void>) | null>(null)

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
        configArea="Knowledge"
        scope="Router-owned knowledge packages, labels, groups, metrics, and signal-facing bindings"
        panelEyebrow="Manager"
        panelTitle="Knowledge Control Plane"
        panelDescription="This surface owns knowledge-base CRUD and the routing-facing resources inside each base. Bases stay as the catalog view, while groups and labels switch to focused paged views."
        pills={[
          {
            label: 'Bases',
            active: activeView === 'bases',
            onClick: () => navigate('/knowledge-bases/bases'),
          },
          {
            label: 'Groups',
            active: activeView === 'groups',
            onClick: () => navigate('/knowledge-bases/groups'),
          },
          {
            label: 'Labels',
            active: activeView === 'labels',
            onClick: () => navigate('/knowledge-bases/labels'),
          },
        ]}
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
