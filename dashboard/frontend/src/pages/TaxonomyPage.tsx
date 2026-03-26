import { useNavigate } from 'react-router-dom'
import { useState } from 'react'
import { useReadonly } from '../contexts/ReadonlyContext'
import ConfigPageManagerLayout from './ConfigPageManagerLayout'
import ConfigPageTaxonomyClassifiers from './ConfigPageTaxonomyClassifiers'
import EditModal, { type EditFormData, type FieldConfig } from '../components/EditModal'
import type { OpenEditModal } from './configPageRouterSectionSupport'

export type KnowledgeBaseView = 'knowledge-bases' | 'groups' | 'labels' | 'exemplars'

interface TaxonomyPageProps {
  activeView: KnowledgeBaseView
}

const VIEW_META: Record<KnowledgeBaseView, { title: string; description: string }> = {
  'knowledge-bases': {
    title: 'Knowledge Bases',
    description: 'Manage built-in and custom embedding KB packages, inspect signal bindings, and curate the active router KB catalog.',
  },
  groups: {
    title: 'Knowledge Base Groups',
    description: 'Review and edit group definitions per KB, keeping higher-level routing semantics aligned with the active label package.',
  },
  labels: {
    title: 'Knowledge Base Labels',
    description: 'Curate label definitions, threshold overrides, and signal-facing label bindings without mixing KB management into Global Config.',
  },
  exemplars: {
    title: 'Knowledge Base Exemplars',
    description: 'Edit exemplar text that grounds KB embedding scores and inspect how each label package is represented on disk.',
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
        eyebrow="Knowledge Base"
        title={meta.title}
        description={meta.description}
        configArea="Knowledge Base"
        scope="Router-owned KB packages, labels, groups, metrics, and exemplar corpora"
        panelEyebrow="Manager"
        panelTitle="Knowledge Base Control Plane"
        panelDescription="This surface owns knowledge base CRUD and the KB-local resources that drive routing. Use the top navbar dropdown or the pills below to move across knowledge base, group, label, and exemplar views."
        pills={[
          {
            label: 'Knowledge Bases',
            active: activeView === 'knowledge-bases',
            onClick: () => navigate('/knowledge-bases/knowledge-bases'),
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
          {
            label: 'Exemplars',
            active: activeView === 'exemplars',
            onClick: () => navigate('/knowledge-bases/exemplars'),
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
