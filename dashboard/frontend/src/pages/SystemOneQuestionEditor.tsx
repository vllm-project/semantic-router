import { motion } from 'framer-motion'
import { ChevronDown, Plus, Trash2, TriangleAlert, X } from 'lucide-react'
import {
  Button,
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuTrigger,
  Field,
  Input,
  cn,
} from '../ui'
import { QUESTION_ICONS } from './decisionQuestionTypes'
import {
  QUESTION_TYPES,
  newQuestion,
  type QuestionDraft,
  type QuestionType,
  type SystemOneQuestion,
} from './systemOnePlayground'

const fieldClass = 'h-8 font-mono text-[12.5px]'

export default function SystemOneQuestionEditor({
  draft,
  index,
  supported,
  duplicate,
  onChange,
  onRemove,
}: {
  draft: QuestionDraft
  index: number
  supported: readonly string[] | undefined
  duplicate: boolean
  onChange: (next: QuestionDraft) => void
  onRemove?: () => void
}) {
  const question = draft.question
  const label = draft.name.trim() || `question ${index + 1}`
  const update = (patch: Partial<SystemOneQuestion>) =>
    onChange({ ...draft, question: { ...question, ...patch } })
  const options = question.type === 'choice' ? question.choices : question.labels
  const optionsField = question.type === 'choice' ? 'choices' : 'labels'
  const noun = question.type === 'choice' ? 'option' : 'label'
  const unsupported = supported !== undefined && !supported.includes(question.type)
  const meta = QUESTION_TYPES.find((item) => item.type === question.type)
  const Icon = QUESTION_ICONS[question.type]
  const switchType = (type: QuestionType) => {
    if (type === question.type) return
    onChange({
      ...draft,
      variants: { ...draft.variants, [question.type]: question },
      question: draft.variants?.[type] ?? newQuestion(type).question,
    })
  }
  return (
    <motion.div
      layout="position"
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, x: 12 }}
      transition={{ duration: 0.25, ease: [0.16, 1, 0.3, 1] }}
      className={cn(
        'group flex flex-col gap-2.5 rounded-lg border bg-card p-3',
        unsupported || duplicate ? 'border-warning/50' : 'border-border',
      )}
    >
      <div className="flex items-center gap-2">
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <button
              type="button"
              aria-label={`Question type for ${label}: ${meta?.label ?? question.type}`}
              className="inline-flex h-6 shrink-0 items-center gap-1 rounded-full bg-chart-3/12 px-2.5 text-xs font-medium text-chart-3 outline-none transition-colors hover:bg-chart-3/20 focus-visible:ring-[3px] focus-visible:ring-ring/40"
            >
              <Icon className="size-3" />
              {meta?.label ?? question.type}
              <ChevronDown className="size-3 opacity-70" />
            </button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="start" className="w-64">
            <DropdownMenuLabel>Question type</DropdownMenuLabel>
            <DropdownMenuRadioGroup
              value={question.type}
              onValueChange={(value) => switchType(value as QuestionType)}
            >
              {QUESTION_TYPES.map((item) => {
                const disabled = supported !== undefined && !supported.includes(item.type)
                return (
                  <DropdownMenuRadioItem key={item.type} value={item.type} disabled={disabled}>
                    <span className="flex flex-col">
                      <span>{item.label}</span>
                      <span className="text-xs text-subtle-foreground">
                        {disabled
                          ? 'This runtime does not support this question type.'
                          : item.detail}
                      </span>
                    </span>
                  </DropdownMenuRadioItem>
                )
              })}
            </DropdownMenuRadioGroup>
          </DropdownMenuContent>
        </DropdownMenu>
        <input
          value={draft.name}
          onChange={(event) => onChange({ ...draft, name: event.target.value })}
          aria-label={`Question ${index + 1} name`}
          placeholder="name"
          spellCheck={false}
          className="h-7 min-w-0 flex-1 rounded-md border border-transparent bg-transparent px-1.5 font-mono text-[12.5px] font-medium outline-none hover:border-border focus:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/40"
        />
        {onRemove ? (
          <Button
            variant="ghost"
            size="icon-xs"
            aria-label={`Remove question ${draft.name}`}
            onClick={onRemove}
            className="opacity-60 group-hover:opacity-100"
          >
            <Trash2 />
          </Button>
        ) : null}
      </div>
      <Input
        value={question.instructions}
        onChange={(event) => update({ instructions: event.target.value })}
        aria-label={`Instructions for ${label}`}
        placeholder="What should the model determine?"
        className="h-8 text-[13px]"
      />
      {options ? (
        <div className="flex flex-col gap-1.5">
          <div className="flex items-center justify-between text-[11px] text-subtle-foreground">
            <span>{question.type === 'choice' ? 'Choices' : 'Labels'}</span>
            <span>Key and optional description</span>
          </div>
          {options.map((option, position) => (
            <div
              key={position}
              className="grid grid-cols-[minmax(0,9rem)_minmax(0,1fr)_auto] gap-1.5"
            >
              <Input
                aria-label={`${label} ${noun} ${position + 1} key`}
                value={option.key}
                placeholder="key"
                spellCheck={false}
                className={fieldClass}
                onChange={(event) =>
                  update({
                    [optionsField]: options.map((item, at) =>
                      at === position ? { ...item, key: event.target.value } : item,
                    ),
                  })
                }
              />
              <Input
                aria-label={`${label} ${noun} ${position + 1} description`}
                value={option.description ?? ''}
                placeholder="Describe this option"
                className="h-8 text-[12.5px]"
                onChange={(event) =>
                  update({
                    [optionsField]: options.map((item, at) =>
                      at === position ? { ...item, description: event.target.value } : item,
                    ),
                  })
                }
              />
              <Button
                variant="ghost"
                size="icon-sm"
                aria-label={`Remove ${noun} ${position + 1}`}
                disabled={options.length <= (question.type === 'choice' ? 2 : 1)}
                onClick={() =>
                  update({ [optionsField]: options.filter((_, at) => at !== position) })
                }
              >
                <X />
              </Button>
            </div>
          ))}
          <Button
            variant="ghost"
            size="xs"
            className="w-fit"
            disabled={options.length >= 255}
            onClick={() =>
              update({
                [optionsField]: [
                  ...options,
                  { key: `${noun}_${options.length + 1}`, description: '' },
                ],
              })
            }
          >
            <Plus />
            Add {noun}
          </Button>
        </div>
      ) : null}
      {question.type === 'score' ? (
        <div className="flex flex-col gap-1.5">
          <div className="flex items-center justify-between text-[11px] text-subtle-foreground">
            <span>Score levels</span>
            <span>Lowest → highest</span>
          </div>
          {(question.levels ?? []).map((level, position) => (
            <div
              key={position}
              className="grid grid-cols-[1.25rem_minmax(0,1fr)_auto] items-center gap-1.5"
            >
              <span className="text-center font-mono text-[11px] text-subtle-foreground tabular">
                {position}
              </span>
              <Input
                aria-label={`${label} level ${position}`}
                value={level}
                className="h-8 text-[12.5px]"
                onChange={(event) =>
                  update({
                    levels: question.levels?.map((item, at) =>
                      at === position ? event.target.value : item,
                    ),
                  })
                }
              />
              <Button
                variant="ghost"
                size="icon-sm"
                aria-label={`Remove level ${position}`}
                disabled={(question.levels?.length ?? 0) <= 2}
                onClick={() =>
                  update({ levels: question.levels?.filter((_, at) => at !== position) })
                }
              >
                <X />
              </Button>
            </div>
          ))}
          <Button
            variant="ghost"
            size="xs"
            className="w-fit"
            disabled={(question.levels?.length ?? 0) >= 10}
            onClick={() => update({ levels: [...(question.levels ?? []), ''] })}
          >
            <Plus />
            Add level
          </Button>
        </div>
      ) : null}
      {question.type === 'noul' ? (
        <p className="text-xs text-subtle-foreground">
          Answered with P(true) from 0 to 1. Ask one clear question; the model supplies its default
          false and true criteria.
        </p>
      ) : null}
      <Collapsible>
        <CollapsibleTrigger className="group/settings inline-flex items-center gap-1 text-xs text-muted-foreground outline-none hover:text-foreground focus-visible:ring-[3px] focus-visible:ring-ring/40">
          <ChevronDown className="size-3.5 transition-transform group-data-[state=open]/settings:rotate-180" />
          Question settings
        </CollapsibleTrigger>
        <CollapsibleContent>
          <div className="mt-2 grid grid-cols-1 gap-3 sm:grid-cols-2">
            <Field label="State field" optional htmlFor={`${draft.id}-over`}>
              <Input
                id={`${draft.id}-over`}
                value={question.over ?? ''}
                onChange={(event) => update({ over: event.target.value })}
                placeholder="Model default · or e.g. request, answer"
                className={fieldClass}
              />
            </Field>
            {question.type === 'set' || question.type === 'span' ? (
              <Field label="Selection threshold" optional htmlFor={`${draft.id}-threshold`}>
                <Input
                  id={`${draft.id}-threshold`}
                  type="number"
                  min={0}
                  max={1}
                  step={0.05}
                  value={question.threshold ?? ''}
                  placeholder="Model default"
                  className={fieldClass}
                  onChange={(event) =>
                    update({
                      threshold: event.target.value === '' ? undefined : event.target.valueAsNumber,
                    })
                  }
                />
              </Field>
            ) : null}
          </div>
          <p className="mt-1.5 text-xs text-subtle-foreground">
            Leave these empty to use the model’s defaults. A span reads one state field.
          </p>
        </CollapsibleContent>
      </Collapsible>
      {unsupported || duplicate ? (
        <p className="flex items-center gap-1.5 text-xs text-warning-foreground">
          <TriangleAlert className="size-3.5 shrink-0" />
          {unsupported
            ? `This runtime does not answer ${meta?.label ?? question.type} questions. Pick another type or target.`
            : 'Question names must be unique.'}
        </p>
      ) : null}
    </motion.div>
  )
}
