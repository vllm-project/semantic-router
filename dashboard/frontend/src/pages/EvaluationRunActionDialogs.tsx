import type { RefObject } from 'react'

import ConfirmDialog from '../components/ConfirmDialog'
import EvaluationIssueDetails from '../components/evaluation-plane/EvaluationIssueDetails'
import type { EvaluationRun } from '../types/evaluationPlane'

interface EvaluationRunActionDialogsProps {
  cancelTarget: EvaluationRun | null
  deleteTarget: EvaluationRun | null
  mutationKey: string | null
  error: string | null
  returnFocusRef: RefObject<HTMLElement | null>
  cancelReturnFocusMode: 'fallback' | 'always'
  deleteReturnFocusMode: 'fallback' | 'always'
  onCloseCancel: () => void
  onCloseDelete: () => void
  onConfirmCancel: () => void | Promise<void>
  onConfirmDelete: () => void | Promise<void>
}

export default function EvaluationRunActionDialogs({
  cancelTarget,
  deleteTarget,
  mutationKey,
  error,
  returnFocusRef,
  cancelReturnFocusMode,
  deleteReturnFocusMode,
  onCloseCancel,
  onCloseDelete,
  onConfirmCancel,
  onConfirmDelete,
}: EvaluationRunActionDialogsProps) {
  const cancelPairID = cancelTarget?.controlled_pair?.pair_id
  const deletePairID = deleteTarget?.controlled_pair?.pair_id
  return (
    <>
      <ConfirmDialog
        isOpen={cancelTarget !== null}
        title={
          cancelPairID
            ? 'Cancel controlled comparison?'
            : `Cancel ${cancelTarget?.name || 'this run'}?`
        }
        description={
          cancelPairID
            ? 'Both runs stop together. Their execution timelines and cancelled status remain available.'
            : 'Execution stops and no completed report is created. The timeline and cancelled status remain available.'
        }
        eyebrow={cancelPairID ? 'Controlled comparison' : 'Evaluation run'}
        confirmLabel={cancelPairID ? 'Cancel comparison' : 'Cancel run'}
        pendingLabel={cancelPairID ? 'Cancelling comparison…' : 'Cancelling…'}
        tone="warning"
        pending={
          mutationKey ===
          (cancelPairID ? `cancel-pair:${cancelPairID}` : `cancel:${cancelTarget?.id || ''}`)
        }
        errorMessage={
          error
            ? cancelPairID
              ? 'The controlled comparison could not be cancelled. Retry or close this dialog.'
              : 'The run could not be cancelled. Retry or close this dialog.'
            : undefined
        }
        errorDetails={
          error ? (
            <EvaluationIssueDetails issues={[{ label: 'Cancellation request', message: error }]} />
          ) : undefined
        }
        returnFocusRef={returnFocusRef}
        returnFocusMode={cancelReturnFocusMode}
        onCancel={onCloseCancel}
        onConfirm={onConfirmCancel}
      />
      <ConfirmDialog
        isOpen={deleteTarget !== null}
        title={
          deletePairID
            ? 'Delete controlled comparison?'
            : `Delete ${deleteTarget?.name || 'this run'}?`
        }
        description={
          deletePairID
            ? 'This permanently removes both runs and their reports from Evaluation. Download anything you need before continuing.'
            : 'This permanently removes the run and its report from Evaluation. Download anything you need before continuing.'
        }
        eyebrow={deletePairID ? 'Controlled comparison' : 'Evaluation run'}
        confirmLabel={deletePairID ? 'Delete comparison' : 'Delete run'}
        pendingLabel={deletePairID ? 'Deleting comparison…' : 'Deleting…'}
        pending={
          mutationKey ===
          (deletePairID ? `delete-pair:${deletePairID}` : `delete:${deleteTarget?.id || ''}`)
        }
        errorMessage={
          error
            ? deletePairID
              ? 'The controlled comparison could not be deleted. Retry or close this dialog.'
              : 'The run could not be deleted. Retry or close this dialog.'
            : undefined
        }
        errorDetails={
          error ? (
            <EvaluationIssueDetails issues={[{ label: 'Deletion request', message: error }]} />
          ) : undefined
        }
        confirmationText={deletePairID ? 'DELETE COMPARISON' : deleteTarget?.name}
        returnFocusRef={returnFocusRef}
        returnFocusMode={deleteReturnFocusMode}
        onCancel={onCloseDelete}
        onConfirm={onConfirmDelete}
      />
    </>
  )
}
