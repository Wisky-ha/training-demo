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
  modelType: string | null
  datasetId: string | null
  preprocessScriptId: string | null
  trainScriptId: string | null
  currentStep: WorkflowStepId
}
