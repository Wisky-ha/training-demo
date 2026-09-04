import type {
  DatasetSplitResult,
  DatasetUploadResult,
  EntityId,
  ModelEvaluation,
  ModelTypeCode,
  ModelVersionSummary,
  PreprocessTask,
  TrainingJob,
} from './contracts'

export const workflowSteps = [
  { id: 'model-type', label: '选择模型类型', caption: 'Model type' },
  { id: 'upload', label: '上传数据', caption: 'Dataset' },
  { id: 'preprocess', label: '数据预处理', caption: 'Preprocess' },
  { id: 'split', label: '数据集划分', caption: 'Split' },
  { id: 'train', label: '模型训练', caption: 'Training' },
  { id: 'evaluate', label: '模型评估', caption: 'Evaluate' },
  { id: 'publish', label: '保存与发布', caption: 'Publish' },
] as const

export type WorkflowStepId = (typeof workflowSteps)[number]['id']

export interface WorkflowDraft {
  modelType: ModelTypeCode | null
  datasetId: EntityId | null
  dataset: DatasetUploadResult | null
  preprocessScriptId: EntityId | null
  preprocessTaskId: EntityId | null
  preprocessTask: PreprocessTask | null
  split: DatasetSplitResult | null
  trainScriptId: EntityId | null
  trainingJobId: EntityId | null
  trainingJob: TrainingJob | null
  evaluation: ModelEvaluation | null
  modelVersion: ModelVersionSummary | null
  currentStep: WorkflowStepId
}
