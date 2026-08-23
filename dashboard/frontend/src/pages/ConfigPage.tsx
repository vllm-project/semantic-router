import React, { useState } from 'react'

import type { FieldConfig, EditFormData } from '../components/EditModal'
import EditModal from '../components/EditModal'
import type { ConfigSection } from '../components/ConfigNav'
import type { ViewPanelAction, ViewSection } from '../components/ViewModal'
import ViewModal from '../components/ViewModal'
import { useAuth } from '../contexts/AuthContext'
import { canManageRouting } from '../utils/accessControl'
import ConfigPageDecisionsSection from './ConfigPageDecisionsSection'
import ConfigPageEntrypointsRecipesSection from './ConfigPageEntrypointsRecipesSection'
import ConfigPageAgentSection from './ConfigPageAgentSection'
import ConfigPageModelsSection from './ConfigPageModelsSection'
import ConfigPageProjectionsSection from './ConfigPageProjectionsSection'
import ConfigPageSignalsSection from './ConfigPageSignalsSection'
import styles from './ConfigPage.module.css'
import { filterViewActionsForMode, type OpenViewModal } from './configPageRouterSectionSupport'
import type { ConfigData, SignalType } from './configPageSupport'

interface ConfigPageProps {
  activeSection?: ConfigSection
}

const removeSignalByName = (config: ConfigData, type: SignalType, targetName: string) => {
  if (!config.signals) config.signals = {}

  switch (type) {
    case 'Keywords':
      config.signals.keywords = (config.signals.keywords || []).filter(
        (signal) => signal.name !== targetName,
      )
      break
    case 'Embeddings':
      config.signals.embeddings = (config.signals.embeddings || []).filter(
        (signal) => signal.name !== targetName,
      )
      break
    case 'Domain':
      config.signals.domains = (config.signals.domains || []).filter(
        (signal) => signal.name !== targetName,
      )
      break
    case 'Preference':
      config.signals.preferences = (config.signals.preferences || []).filter(
        (signal) => signal.name !== targetName,
      )
      break
    case 'Fact Check':
      config.signals.fact_check = (config.signals.fact_check || []).filter(
        (signal) => signal.name !== targetName,
      )
      break
    case 'User Feedback':
      config.signals.user_feedbacks = (config.signals.user_feedbacks || []).filter(
        (signal) => signal.name !== targetName,
      )
      break
    case 'Reask':
      config.signals.reasks = (config.signals.reasks || []).filter(
        (signal) => signal.name !== targetName,
      )
      break
    case 'Language':
      config.signals.language = (config.signals.language || []).filter(
        (signal) => signal.name !== targetName,
      )
      break
    case 'Context':
      config.signals.context = (config.signals.context || []).filter(
        (signal) => signal.name !== targetName,
      )
      break
    case 'Structure':
      config.signals.structure = (config.signals.structure || []).filter(
        (signal) => signal.name !== targetName,
      )
      break
    case 'Complexity':
      config.signals.complexity = (config.signals.complexity || []).filter(
        (signal) => signal.name !== targetName,
      )
      break
    case 'Modality':
      config.signals.modality = (config.signals.modality || []).filter(
        (signal) => signal.name !== targetName,
      )
      break
    case 'Authz':
      config.signals.role_bindings = (config.signals.role_bindings || []).filter(
        (signal) => signal.name !== targetName,
      )
      break
    case 'Jailbreak':
      config.signals.jailbreak = (config.signals.jailbreak || []).filter(
        (signal) => signal.name !== targetName,
      )
      break
    case 'PII':
      config.signals.pii = (config.signals.pii || []).filter((signal) => signal.name !== targetName)
      break
    case 'KB':
      config.signals.kb = (config.signals.kb || []).filter((signal) => signal.name !== targetName)
      break
    case 'Metadata':
      config.signals.metadata = (config.signals.metadata || []).filter(
        (signal) => signal.name !== targetName,
      )
      break
    case 'Classifier':
      config.signals.classifiers = (config.signals.classifiers || []).filter(
        (signal) => signal.name !== targetName,
      )
      break
  }
}

const removeDecisionByName = (config: ConfigData, targetName: string) => {
  config.decisions = (config.decisions || []).filter((decision) => decision.name !== targetName)
}

const ConfigPage: React.FC<ConfigPageProps> = ({ activeSection = 'models' }) => {
  const { user } = useAuth()
  const routingReadonly = !canManageRouting(user)
  const [editModalOpen, setEditModalOpen] = useState(false)
  const [editModalTitle, setEditModalTitle] = useState('')
  const [editModalData, setEditModalData] = useState<EditFormData | null>(null)
  const [editModalFields, setEditModalFields] = useState<FieldConfig[]>([])
  const [editModalMode, setEditModalMode] = useState<'edit' | 'add'>('edit')
  const [editModalCallback, setEditModalCallback] = useState<
    ((data: EditFormData) => Promise<void>) | null
  >(null)
  const [viewModalOpen, setViewModalOpen] = useState(false)
  const [viewModalTitle, setViewModalTitle] = useState('')
  const [viewModalSections, setViewModalSections] = useState<ViewSection[]>([])
  const [viewModalEditCallback, setViewModalEditCallback] = useState<(() => void) | null>(null)
  const [viewModalActions, setViewModalActions] = useState<ViewPanelAction[]>([])
  const [decisionsSearch, setDecisionsSearch] = useState('')
  const [signalsSearch, setSignalsSearch] = useState('')
  const [modelsSearch, setModelsSearch] = useState('')

  const openEditModal = <TForm extends object>(
    title: string,
    data: TForm,
    fields: FieldConfig<TForm>[],
    callback: (data: TForm) => Promise<void>,
    mode: 'edit' | 'add' = 'edit',
  ) => {
    setEditModalTitle(title)
    setEditModalData(data as EditFormData)
    setEditModalFields(fields as FieldConfig[])
    setEditModalMode(mode)
    setEditModalCallback(() => async (rawData: EditFormData) => callback(rawData as TForm))
    setEditModalOpen(true)
  }

  const openViewModal: OpenViewModal = (title, sections, onEdit, actions = []) => {
    setViewModalTitle(title)
    setViewModalSections(sections)
    setViewModalEditCallback(() => onEdit || null)
    setViewModalActions(
      actions.map((action) => ({
        ...action,
        onClick: () => {
          setViewModalOpen(false)
          action.onClick()
        },
      })),
    )
    setViewModalOpen(true)
  }

  const activePanel = (() => {
    switch (activeSection) {
      case 'signals':
        return (
          <ConfigPageSignalsSection
            isReadonly={routingReadonly}
            signalsSearch={signalsSearch}
            onSignalsSearchChange={setSignalsSearch}
            openEditModal={openEditModal}
            openViewModal={openViewModal}
            removeSignalByName={removeSignalByName}
          />
        )
      case 'projections':
        return (
          <ConfigPageProjectionsSection
            isReadonly={routingReadonly}
            openEditModal={openEditModal}
            openViewModal={openViewModal}
          />
        )
      case 'decisions':
        return (
          <ConfigPageDecisionsSection
            isReadonly={routingReadonly}
            decisionsSearch={decisionsSearch}
            onDecisionsSearchChange={setDecisionsSearch}
            openEditModal={openEditModal}
            openViewModal={openViewModal}
            removeDecisionByName={removeDecisionByName}
          />
        )
      case 'entrypoints-recipes':
        return <ConfigPageEntrypointsRecipesSection />
      case 'agent':
        return <ConfigPageAgentSection />
      case 'models':
      default:
        return (
          <ConfigPageModelsSection
            isReadonly={routingReadonly}
            canVerifyModels={canManageRouting(user)}
            modelsSearch={modelsSearch}
            onModelsSearchChange={setModelsSearch}
            openEditModal={openEditModal}
            openViewModal={openViewModal}
          />
        )
    }
  })()

  return (
    <div className={styles.container}>
      <div className={styles.content}>
        <div className={styles.contentArea}>{activePanel}</div>
      </div>

      <EditModal
        isOpen={editModalOpen}
        onClose={() => setEditModalOpen(false)}
        onSave={editModalCallback || (async () => {})}
        title={editModalTitle}
        data={editModalData}
        fields={editModalFields}
        mode={editModalMode}
      />
      <ViewModal
        isOpen={viewModalOpen}
        onClose={() => setViewModalOpen(false)}
        onEdit={routingReadonly ? undefined : viewModalEditCallback || undefined}
        title={viewModalTitle}
        sections={viewModalSections}
        actions={filterViewActionsForMode(viewModalActions, routingReadonly)}
      />
    </div>
  )
}

export default ConfigPage
