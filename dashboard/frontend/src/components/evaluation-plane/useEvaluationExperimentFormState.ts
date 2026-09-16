import { useRef, useState } from 'react'

import type {
  EvaluationChangeProfileId,
  EvaluationMode,
  EvaluationTrackId,
} from '../../types/evaluationPlane'

export interface EvaluationCapacitySLOInput {
  requiredConcurrency: string
  maxLatencyP95MS: string
  maxErrorRate: string
  minThroughputRPS: string
  minThroughputScalingEfficiency: string
}

export const EMPTY_CAPACITY_SLO_INPUT: EvaluationCapacitySLOInput = {
  requiredConcurrency: '',
  maxLatencyP95MS: '',
  maxErrorRate: '',
  minThroughputRPS: '',
  minThroughputScalingEfficiency: '',
}

interface EvaluationExperimentFormInitialState {
  mode: EvaluationMode
  changeProfile: EvaluationChangeProfileId | ''
  targetID: string
  suiteIDs: string[]
  trackIDs: EvaluationTrackId[]
  canAutoStart: boolean
}

export function useEvaluationExperimentFormState(initial: EvaluationExperimentFormInitialState) {
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [mode, setMode] = useState<EvaluationMode>(initial.mode)
  const [changeProfile, setChangeProfile] = useState<EvaluationChangeProfileId | ''>(
    initial.changeProfile,
  )
  const [targetID, setTargetID] = useState(initial.targetID)
  const [suiteIDs, setSuiteIDs] = useState<string[]>(initial.suiteIDs)
  const [trackIDs, setTrackIDs] = useState<EvaluationTrackId[]>(initial.trackIDs)
  const [sampleLimit, setSampleLimit] = useState(100)
  const [concurrency, setConcurrency] = useState(4)
  const [capacitySLOInput, setCapacitySLOInput] =
    useState<EvaluationCapacitySLOInput>(EMPTY_CAPACITY_SLO_INPUT)
  const [seed, setSeed] = useState(42)
  const [baselineRunID, setBaselineRunID] = useState('')
  const [autoStart, setAutoStart] = useState(initial.canAutoStart)
  const [validationError, setValidationError] = useState('')
  const errorRef = useRef<HTMLDivElement | null>(null)
  const createAttempt = useRef<{ fingerprint: string; id: string } | null>(null)

  return {
    name,
    setName,
    description,
    setDescription,
    mode,
    setMode,
    changeProfile,
    setChangeProfile,
    targetID,
    setTargetID,
    suiteIDs,
    setSuiteIDs,
    trackIDs,
    setTrackIDs,
    sampleLimit,
    setSampleLimit,
    concurrency,
    setConcurrency,
    capacitySLOInput,
    setCapacitySLOInput,
    seed,
    setSeed,
    baselineRunID,
    setBaselineRunID,
    autoStart,
    setAutoStart,
    validationError,
    setValidationError,
    errorRef,
    createAttempt,
  }
}

export type EvaluationExperimentFormState = ReturnType<typeof useEvaluationExperimentFormState>
