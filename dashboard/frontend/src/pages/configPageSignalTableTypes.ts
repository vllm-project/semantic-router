import type {
  ClassifierSignal,
  ComplexitySignal,
  ContextSignal,
  ConversationSignal,
  DomainSignal,
  EmbeddingSignal,
  FactCheckSignal,
  JailbreakSignal,
  KBSignal,
  KeywordSignal,
  LanguageSignal,
  MetadataSignal,
  ModalitySignal,
  PIISignal,
  PreferenceSignal,
  ReaskSignal,
  RoleBindingSignal,
  SignalType,
  StructureSignal,
  UserFeedbackSignal,
} from './configPageSupport'

type UnifiedSignalData = Partial<
  KeywordSignal &
    EmbeddingSignal &
    DomainSignal &
    PreferenceSignal &
    FactCheckSignal &
    UserFeedbackSignal &
    ReaskSignal &
    LanguageSignal &
    ContextSignal &
    StructureSignal &
    ComplexitySignal &
    ModalitySignal &
    RoleBindingSignal &
    JailbreakSignal &
    PIISignal &
    KBSignal &
    MetadataSignal &
    ClassifierSignal &
    ConversationSignal
>

export interface UnifiedSignal {
  name: string
  type: SignalType
  summary: string
  rawData: UnifiedSignalData
}
