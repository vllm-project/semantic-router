import { useNavigate } from 'react-router-dom'
import { useState } from 'react'
import { useReadonly } from '../contexts/ReadonlyContext'
import ConfigPageManagerLayout from './ConfigPageManagerLayout'
import ConfigPageTaxonomyClassifiers from './ConfigPageTaxonomyClassifiers'
import EditModal, { type EditFormData, type FieldConfig } from '../components/EditModal'
import type { OpenEditModal } from './configPageRouterSectionSupport'

export type TaxonomyView = 'classifiers' | 'tiers' | 'categories' | 'exemplars'

interface TaxonomyPageProps {
  activeView: TaxonomyView
}

const VIEW_META: Record<TaxonomyView, { title: string; description: string }> = {
  classifiers: {
    title: 'Taxonomy Classifiers',
    description: 'Manage built-in and custom taxonomy classifier packages, inspect signal bindings, and curate the router taxonomy catalog.',
  },
  tiers: {
    title: 'Taxonomy Tiers',
    description: 'Review and edit tier definitions per classifier, keeping tier-level routing semantics aligned with the active taxonomy package.',
  },
  categories: {
    title: 'Taxonomy Categories',
    description: 'Curate category definitions, tier assignments, and category-level bindings without mixing taxonomy management into Global Config.',
  },
  exemplars: {
    title: 'Taxonomy Exemplars',
    description: 'Edit the exemplar text that powers classifier embeddings and inspect how each category package is grounded on disk.',
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
        eyebrow="Taxonomy"
        title={meta.title}
        description={meta.description}
        configArea="Taxonomy"
        scope="Classifier packages, tiers, categories, and exemplar corpora"
        panelEyebrow="Manager"
        panelTitle="Taxonomy Control Plane"
        panelDescription="This surface owns taxonomy classifier CRUD and the classifier-local resources that drive routing. Use the top navbar dropdown or the pills below to move across classifier, tier, category, and exemplar views."
        pills={[
          {
            label: 'Classifiers',
            active: activeView === 'classifiers',
            onClick: () => navigate('/taxonomy/classifiers'),
          },
          {
            label: 'Tiers',
            active: activeView === 'tiers',
            onClick: () => navigate('/taxonomy/tiers'),
          },
          {
            label: 'Categories',
            active: activeView === 'categories',
            onClick: () => navigate('/taxonomy/categories'),
          },
          {
            label: 'Exemplars',
            active: activeView === 'exemplars',
            onClick: () => navigate('/taxonomy/exemplars'),
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
