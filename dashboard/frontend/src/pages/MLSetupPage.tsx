import React, { useRef, useState } from 'react';
import styles from './MLSetupPage.module.css';
import { ObjectListEditor, type ObjectEditorField } from '../components/ObjectListEditor';
import { StringListEditor } from '../components/StringListEditor';
import { useMLPipelineWizard } from '../hooks/useMLPipeline';
import {
  ML_ALGORITHM_INFO,
  PIPELINE_STEPS,
  type MLAlgorithm,
  type SvmKernel,
} from '../types/mlPipeline';
import { getDownloadUrl } from '../utils/mlPipelineApi';
import {
  getDecisionEntriesError,
  getMlpHiddenLayersError,
  parseMlpHiddenLayers,
  serializeMlpHiddenLayers,
  type HiddenLayerEntry,
} from './mlSetupStructuredFieldSupport';
import MLSetupBenchmarkStep from './MLSetupBenchmarkStep';
import MLSetupProgressDisplay from './MLSetupProgressDisplay';

const HIDDEN_LAYER_FIELDS: ObjectEditorField<HiddenLayerEntry>[] = [
  {
    key: 'size',
    label: 'Layer width',
    type: 'number',
    min: 1,
    step: 1,
    required: true,
    placeholder: '256',
  },
];

const MLSetupPage: React.FC = () => {
  const wizard = useMLPipelineWizard();

  return (
    <div className={styles.page}>
      {/* Header */}
      <div className={styles.header}>
        <h1 className={styles.title}>ML Model Selection Setup</h1>
        <p className={styles.subtitle}>
          Configure intelligent model routing by benchmarking your models, training ML classifiers,
          and generating deployment-ready configuration.
        </p>
      </div>

      {/* Stepper */}
      <div className={styles.stepper}>
        {PIPELINE_STEPS.map((step, idx) => (
          <React.Fragment key={step.key}>
            {idx > 0 && (
              <div
                className={`${styles.stepConnector} ${
                  idx <= wizard.currentStep ? styles.stepConnectorActive : ''
                }`}
              />
            )}
            <button
              className={`${styles.stepItem} ${
                idx === wizard.currentStep ? styles.stepActive : ''
              } ${
                idx < wizard.currentStep ? styles.stepCompleted : ''
              }`}
              onClick={() => wizard.goToStep(idx)}
            >
              <div className={styles.stepCircle}>
                {idx < wizard.currentStep ? (
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                    <polyline points="20 6 9 17 4 12" />
                  </svg>
                ) : (
                  idx + 1
                )}
              </div>
              <span className={styles.stepLabel}>{step.label}</span>
            </button>
          </React.Fragment>
        ))}
      </div>

      {/* Error alert */}
      {wizard.actionError && (
        <div className={styles.errorAlert}>
          <span>{wizard.actionError}</span>
          <button className={styles.errorDismiss} onClick={wizard.clearError}>
            &times;
          </button>
        </div>
      )}

      {/* Step content */}
      {wizard.currentStep === 0 && <MLSetupBenchmarkStep wizard={wizard} />}
      {wizard.currentStep === 1 && <TrainStep wizard={wizard} />}
      {wizard.currentStep === 2 && <ConfigStep wizard={wizard} />}

      {/* Navigation */}
      <div className={styles.actions}>
        <div className={styles.actionsLeft}>
          {wizard.currentStep > 0 && (
            <button className={styles.btnSecondary} onClick={wizard.prevStep}>
              Back
            </button>
          )}
          <button className={styles.btnSecondary} onClick={wizard.reset}>
            Reset
          </button>
        </div>
        <div className={styles.actionsRight}>
          {wizard.currentStep < 2 && (
            <button
              className={styles.btnPrimary}
              onClick={wizard.nextStep}
              disabled={!wizard.canAdvanceToStep(wizard.currentStep + 1)}
            >
              Next Step
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

interface StepProps {
  wizard: ReturnType<typeof useMLPipelineWizard>;
}


/* ============================================================
   Step 1: Train
   ============================================================ */
const TrainStep: React.FC<StepProps> = ({ wizard }) => {
  const trainingDataInputRef = useRef<HTMLInputElement>(null);
  const jobDone = wizard.trainProgress.completed && wizard.trainProgress.job?.status === 'completed';
  const jobFailed = wizard.trainProgress.job?.status === 'failed';

  const algorithms: MLAlgorithm[] = ['knn', 'kmeans', 'svm', 'mlp'];
  const hasDataSource = wizard.benchmarkJobId !== null || wizard.trainingDataFile !== null;
  const hiddenLayersError = wizard.selectedAlgorithms.includes('mlp')
    ? getMlpHiddenLayersError(wizard.mlpHiddenSizes)
    : null;

  return (
    <div className={styles.stepContent}>
      <h2 className={styles.stepTitle}>Step 2: Train ML Classifiers</h2>
      <p className={styles.stepDescription}>
        Select the ML algorithms to train on your benchmark data. Each algorithm will produce a
        model that can be used for intelligent request routing.
      </p>

      {/* Training data source section */}
      {wizard.benchmarkJobId ? (
        <div className={styles.dataSourceBanner}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#22c55e" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="10" />
            <polyline points="16 8 10 16 7 13" />
          </svg>
          <span>Using benchmark output from job <code>{wizard.benchmarkJobId}</code></span>
        </div>
      ) : (
        <div className={styles.formGroup}>
          <label className={styles.formLabel}>Training Data (JSONL)</label>
          <p className={styles.formHint} style={{ marginBottom: '0.5rem' }}>
            Upload your existing benchmark / training data file to skip the benchmark step.
          </p>
          <input
            ref={trainingDataInputRef}
            type="file"
            accept=".jsonl,.json"
            className={styles.fileInput}
            onChange={(e) => wizard.setTrainingDataFile(e.target.files?.[0] || null)}
          />
          <div
            className={`${styles.fileDropZone} ${wizard.trainingDataFile ? styles.fileDropZoneActive : ''}`}
            onClick={() => trainingDataInputRef.current?.click()}
          >
            {wizard.trainingDataFile ? (
              <div className={styles.fileDropName}>{wizard.trainingDataFile.name}</div>
            ) : (
              <>
                <div className={styles.fileDropIcon}>
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                    <polyline points="17 8 12 3 7 8" />
                    <line x1="12" y1="3" x2="12" y2="15" />
                  </svg>
                </div>
                <div className={styles.fileDropText}>Click to upload training data JSONL</div>
              </>
            )}
          </div>
        </div>
      )}

      {/* Algorithm selection */}
      <div className={styles.formGroup}>
        <label className={styles.formLabel}>Algorithms</label>
        <div className={styles.algorithmGrid}>
          {algorithms.map((alg) => {
            const info = ML_ALGORITHM_INFO[alg];
            const selected = wizard.selectedAlgorithms.includes(alg);
            return (
              <div
                key={alg}
                className={`${styles.algorithmCard} ${selected ? styles.algorithmCardSelected : ''}`}
                onClick={() => wizard.toggleAlgorithm(alg)}
              >
                <div className={styles.algorithmHeader}>
                  <div className={styles.algorithmDot} style={{ background: info.color }} />
                  <span className={styles.algorithmName}>{info.label}</span>
                </div>
                <div className={styles.algorithmDesc}>{info.description}</div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Device selection — only relevant for MLP (PyTorch) */}
      {wizard.selectedAlgorithms.includes('mlp') && (
        <div className={styles.formGroup}>
          <label className={styles.formLabel}>Device</label>
          <div className={styles.deviceSelector}>
            <button
              className={`${styles.deviceOption} ${wizard.device === 'cpu' ? styles.deviceOptionSelected : ''}`}
              onClick={() => wizard.setDevice('cpu')}
            >
              CPU
            </button>
            <button
              className={`${styles.deviceOption} ${wizard.device === 'cuda' ? styles.deviceOptionSelected : ''}`}
              onClick={() => wizard.setDevice('cuda')}
            >
              CUDA (GPU)
            </button>
          </div>
        </div>
      )}

      {/* Advanced Training Settings */}
      <TrainAdvancedSettings wizard={wizard} />

      {hiddenLayersError && <div className={styles.errorAlert}>{hiddenLayersError}</div>}

      {/* Run button or progress */}
      {!wizard.trainJobId && (
        <button
          className={styles.btnPrimary}
          onClick={() => {
            if (!hiddenLayersError) void wizard.startTraining();
          }}
          disabled={
            wizard.actionLoading ||
            wizard.selectedAlgorithms.length === 0 ||
            !hasDataSource ||
            Boolean(hiddenLayersError)
          }
        >
          {wizard.actionLoading ? 'Starting...' : 'Train Models'}
        </button>
      )}

      {wizard.trainJobId && (
        <MLSetupProgressDisplay
          progress={wizard.trainProgress.progress}
          job={wizard.trainProgress.job}
          completed={wizard.trainProgress.completed}
        />
      )}

      {jobDone && (
        <div className={styles.successCard}>
          <div className={styles.successIcon}>
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#22c55e" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10" />
              <polyline points="16 8 10 16 7 13" />
            </svg>
          </div>
          <div className={styles.successTitle}>Training Complete</div>
          <div className={styles.successMessage}>
            {wizard.trainProgress.job?.output_files?.length || 0} model(s) trained successfully.
            Proceed to generate deployment configuration, or train additional algorithms.
          </div>
          <button
            className={styles.btnSecondary}
            onClick={wizard.resetTraining}
            style={{ marginTop: '0.75rem' }}
          >
            Train Again (different algorithms)
          </button>
        </div>
      )}

      {jobFailed && (
        <div className={styles.errorAlert}>
          <span>Training failed: {wizard.trainProgress.job?.error || 'Unknown error'}</span>
          <button
            className={styles.btnSecondary}
            onClick={wizard.resetTraining}
            style={{ marginLeft: '0.75rem', padding: '0.25rem 0.75rem', fontSize: '0.75rem' }}
          >
            Retry
          </button>
        </div>
      )}
    </div>
  );
};

/* ============================================================
   Step 2: Config
   ============================================================ */
const ConfigStep: React.FC<StepProps> = ({ wizard }) => {
  const jobDone = wizard.configProgress.completed && wizard.configProgress.job?.status === 'completed';
  const jobFailed = wizard.configProgress.job?.status === 'failed';
  const algorithms: MLAlgorithm[] = ['knn', 'kmeans', 'svm', 'mlp'];
  const decisionsError = getDecisionEntriesError(wizard.decisions);

  return (
    <div className={styles.stepContent}>
      <h2 className={styles.stepTitle}>Step 3: Generate Configuration</h2>
      <p className={styles.stepDescription}>
        Define your routing decisions and model references. This will generate a deployment-ready
        YAML configuration file for semantic-router.
      </p>

      {/* Models path */}
      <div className={styles.formGroup}>
        <label className={styles.formLabel}>Models Path</label>
        <input
          type="text"
          className={styles.textInput}
          value={wizard.modelsPath}
          onChange={(e) => wizard.setModelsPath(e.target.value)}
          placeholder="/tmp/ml-models"
        />
        <div className={styles.formHint}>Directory where trained ML model files will be stored</div>
      </div>

      {/* Decisions */}
      <div className={styles.formGroup}>
        <label className={styles.formLabel}>Routing Decisions</label>
        <div className={styles.decisionList}>
          {wizard.decisions.map((dec, idx) => (
            <div key={idx} className={styles.decisionCard}>
              <div className={styles.decisionHeader}>
                <span className={styles.decisionNumber}>Decision #{idx + 1}</span>
                {wizard.decisions.length > 1 && (
                  <button className={styles.removeBtn} onClick={() => wizard.removeDecision(idx)}>
                    Remove
                  </button>
                )}
              </div>
              <div className={styles.decisionFields}>
                <div className={styles.formGroup}>
                  <label className={styles.formLabel}>Name</label>
                  <input
                    type="text"
                    className={styles.textInput}
                    value={dec.name}
                    onChange={(e) => wizard.updateDecision(idx, { name: e.target.value })}
                    placeholder="e.g. coding-router"
                  />
                </div>
                <div className={styles.formGroup}>
                  <label className={styles.formLabel}>Algorithm</label>
                  <select
                    className={styles.selectInput}
                    value={dec.algorithm}
                    onChange={(e) =>
                      wizard.updateDecision(idx, { algorithm: e.target.value as MLAlgorithm })
                    }
                  >
                    {algorithms.map((alg) => (
                      <option key={alg} value={alg}>
                        {ML_ALGORITHM_INFO[alg].label}
                      </option>
                    ))}
                  </select>
                </div>
                <div className={styles.formGroup}>
                  <label className={styles.formLabel}>Priority</label>
                  <input
                    type="number"
                    className={styles.textInput}
                    value={dec.priority}
                    onChange={(e) =>
                      wizard.updateDecision(idx, { priority: parseInt(e.target.value) || 0 })
                    }
                    min={0}
                    max={1000}
                    placeholder="100"
                  />
                </div>
                <div className={styles.formGroup}>
                  <span className={styles.formLabel}>Domains</span>
                  <StringListEditor
                    value={dec.domains}
                    onChange={(domains) => wizard.updateDecision(idx, { domains })}
                    itemLabel="Domain"
                    addLabel="Add domain"
                    emptyLabel="No domain constraints."
                    placeholder="e.g. coding, math, general"
                  />
                </div>
                <div className={styles.formGroup} style={{ gridColumn: '1 / -1' }}>
                  <span className={styles.formLabel}>Model Names</span>
                  <StringListEditor
                    value={dec.model_names}
                    onChange={(model_names) => wizard.updateDecision(idx, { model_names })}
                    itemLabel="Model"
                    addLabel="Add model"
                    emptyLabel="Add at least one model reference."
                    placeholder="e.g. codellama-34b"
                  />
                </div>
              </div>
            </div>
          ))}
        </div>
        <button className={styles.addBtn} onClick={wizard.addDecision}>
          + Add Decision
        </button>
      </div>

      {/* Generate button or progress */}
      {decisionsError && <div className={styles.errorAlert}>{decisionsError}</div>}
      {!wizard.configJobId && (
        <button
          className={styles.btnSuccess}
          onClick={() => {
            if (!decisionsError) void wizard.startConfigGeneration();
          }}
          disabled={wizard.actionLoading || Boolean(decisionsError)}
        >
          {wizard.actionLoading ? 'Generating...' : 'Generate Config'}
        </button>
      )}

      {wizard.configJobId && (
        <MLSetupProgressDisplay
          progress={wizard.configProgress.progress}
          job={wizard.configProgress.job}
          completed={wizard.configProgress.completed}
        />
      )}

      {jobDone && wizard.configJobId && (
        <div className={styles.successCard}>
          <div className={styles.successIcon}>
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#22c55e" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10" />
              <polyline points="16 8 10 16 7 13" />
            </svg>
          </div>
          <div className={styles.successTitle}>Configuration Generated</div>
          <div className={styles.successMessage}>
            Your deployment-ready configuration file is ready for download.
          </div>
          <a
            href={getDownloadUrl(wizard.configJobId)}
            className={styles.downloadBtn}
            download
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="7 10 12 15 17 10" />
              <line x1="12" y1="15" x2="12" y2="3" />
            </svg>
            Download YAML Config
          </a>
        </div>
      )}

      {jobFailed && (
        <div className={styles.errorAlert}>
          <span>Config generation failed: {wizard.configProgress.job?.error || 'Unknown error'}</span>
        </div>
      )}
    </div>
  );
};


/* ============================================================
   Advanced Settings: Training
   ============================================================ */
const TrainAdvancedSettings: React.FC<StepProps> = ({ wizard }) => {
  const [open, setOpen] = useState(false);
  const showKnn = wizard.selectedAlgorithms.includes('knn');
  const showKmeans = wizard.selectedAlgorithms.includes('kmeans');
  const showSvm = wizard.selectedAlgorithms.includes('svm');
  const showMlp = wizard.selectedAlgorithms.includes('mlp');

  return (
    <div className={styles.advancedSection}>
      <button className={styles.advancedToggle} onClick={() => setOpen(!open)}>
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
          style={{ transform: open ? 'rotate(90deg)' : 'rotate(0)', transition: 'transform 0.2s' }}>
          <polyline points="9 18 15 12 9 6" />
        </svg>
        Advanced Settings
      </button>
      {open && (
        <div className={styles.advancedContent}>
          {/* General */}
          <div className={styles.advancedGroupTitle}>General</div>
          <div className={styles.advancedGrid}>
            <div className={styles.formGroup}>
              <label className={styles.formLabel}>Quality Weight</label>
              <input type="number" className={styles.numberInput} min={0} max={1} step={0.05}
                value={wizard.qualityWeight}
                onChange={(e) => wizard.setQualityWeight(parseFloat(e.target.value) || 0.9)}
              />
              <div className={styles.formHint}>Weight for quality vs latency (0-1)</div>
            </div>
            <div className={styles.formGroup}>
              <label className={styles.formLabel}>Batch Size</label>
              <input type="number" className={styles.numberInput} min={1} max={256}
                value={wizard.batchSize}
                onChange={(e) => wizard.setBatchSize(parseInt(e.target.value) || 32)}
              />
              <div className={styles.formHint}>Embedding generation batch size</div>
            </div>
          </div>

          {/* KNN */}
          {showKnn && (
            <>
              <div className={styles.advancedGroupTitle}>KNN Parameters</div>
              <div className={styles.advancedGrid}>
                <div className={styles.formGroup}>
                  <label className={styles.formLabel}>K (neighbors)</label>
                  <input type="number" className={styles.numberInput} min={1} max={50}
                    value={wizard.knnK}
                    onChange={(e) => wizard.setKnnK(parseInt(e.target.value) || 5)}
                  />
                </div>
              </div>
            </>
          )}

          {/* KMeans */}
          {showKmeans && (
            <>
              <div className={styles.advancedGroupTitle}>KMeans Parameters</div>
              <div className={styles.advancedGrid}>
                <div className={styles.formGroup}>
                  <label className={styles.formLabel}>Clusters</label>
                  <input type="number" className={styles.numberInput} min={2} max={64}
                    value={wizard.kmeansClusters}
                    onChange={(e) => wizard.setKmeansClusters(parseInt(e.target.value) || 8)}
                  />
                </div>
              </div>
            </>
          )}

          {/* SVM */}
          {showSvm && (
            <>
              <div className={styles.advancedGroupTitle}>SVM Parameters</div>
              <div className={styles.advancedGrid}>
                <div className={styles.formGroup}>
                  <label className={styles.formLabel}>Kernel</label>
                  <select className={styles.selectInput} value={wizard.svmKernel}
                    onChange={(e) => wizard.setSvmKernel(e.target.value as SvmKernel)}>
                    <option value="rbf">RBF (Radial Basis Function)</option>
                    <option value="linear">Linear</option>
                  </select>
                </div>
                <div className={styles.formGroup}>
                  <label className={styles.formLabel}>Gamma</label>
                  <input type="number" className={styles.numberInput} min={0.001} max={100} step={0.1}
                    value={wizard.svmGamma}
                    onChange={(e) => wizard.setSvmGamma(parseFloat(e.target.value) || 1.0)}
                  />
                  <div className={styles.formHint}>RBF kernel parameter</div>
                </div>
              </div>
            </>
          )}

          {/* MLP */}
          {showMlp && (
            <>
              <div className={styles.advancedGroupTitle}>MLP Parameters</div>
              <div className={styles.advancedGrid}>
                <div className={styles.formGroup}>
                  <span className={styles.formLabel}>Hidden Layers</span>
                  <ObjectListEditor
                    value={parseMlpHiddenLayers(wizard.mlpHiddenSizes)}
                    onChange={(values) =>
                      wizard.setMlpHiddenSizes(serializeMlpHiddenLayers(values))
                    }
                    fields={HIDDEN_LAYER_FIELDS}
                    createItem={() => ({ size: 128 })}
                    addLabel="Add hidden layer"
                    emptyLabel="Add at least one hidden layer."
                    itemLabel={(item, index) =>
                      item.size ? `Layer ${index + 1} · ${item.size} units` : `Layer ${index + 1}`
                    }
                    validateItem={(item) =>
                      Number.isInteger(item.size) && (item.size ?? 0) > 0
                        ? []
                        : ['Layer width must be a positive integer.']
                    }
                  />
                  <div className={styles.formHint}>Layer order follows the list from input to output.</div>
                </div>
                <div className={styles.formGroup}>
                  <label className={styles.formLabel}>Epochs</label>
                  <input type="number" className={styles.numberInput} min={1} max={1000}
                    value={wizard.mlpEpochs}
                    onChange={(e) => wizard.setMlpEpochs(parseInt(e.target.value) || 100)}
                  />
                </div>
                <div className={styles.formGroup}>
                  <label className={styles.formLabel}>Learning Rate</label>
                  <input type="number" className={styles.numberInput} min={0.0001} max={1} step={0.0001}
                    value={wizard.mlpLearningRate}
                    onChange={(e) => wizard.setMlpLearningRate(parseFloat(e.target.value) || 0.001)}
                  />
                </div>
                <div className={styles.formGroup}>
                  <label className={styles.formLabel}>Dropout</label>
                  <input type="number" className={styles.numberInput} min={0} max={0.9} step={0.05}
                    value={wizard.mlpDropout}
                    onChange={(e) => wizard.setMlpDropout(parseFloat(e.target.value) || 0.1)}
                  />
                </div>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
};


export default MLSetupPage;
