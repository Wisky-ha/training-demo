import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { WorkflowDraft, WorkflowStepId } from '../types/workflow'

export type Theme = 'light' | 'dark'

interface AppState {
  theme: Theme
  sidebarCollapsed: boolean
  workflow: WorkflowDraft
  toggleTheme: () => void
  setTheme: (theme: Theme) => void
  toggleSidebar: () => void
  setWorkflowStep: (step: WorkflowStepId) => void
  setWorkflowContext: (context: Partial<WorkflowDraft>) => void
  setModelType: (modelType: WorkflowDraft['modelType']) => void
  resetWorkflow: () => void
}

/**
 * Resource objects in this store are deliberately replaceable render caches.
 * The persisted shape below contains IDs only; WorkflowPage re-reads those IDs
 * after a reload before it treats a cached object as current.
 */
const emptyWorkflow = (): WorkflowDraft => ({
  modelType: null,
  datasetId: null,
  dataset: null,
  preprocessScriptId: null,
  preprocessTaskId: null,
  preprocessTask: null,
  splitId: null,
  split: null,
  trainScriptId: null,
  trainingJobId: null,
  trainingJob: null,
  modelVersionId: null,
  evaluation: null,
  modelVersion: null,
  currentStep: 'model-type',
})

const hasOwn = (value: object, key: PropertyKey): boolean =>
  Object.prototype.hasOwnProperty.call(value, key)

const persistedWorkflow = (workflow: WorkflowDraft) => ({
  modelType: workflow.modelType,
  datasetId: workflow.datasetId,
  preprocessScriptId: workflow.preprocessScriptId,
  preprocessTaskId: workflow.preprocessTaskId,
  splitId: workflow.splitId,
  trainScriptId: workflow.trainScriptId,
  trainingJobId: workflow.trainingJobId,
  modelVersionId: workflow.modelVersionId,
  currentStep: workflow.currentStep,
})

export const useAppStore = create<AppState>()(
  persist(
    (set) => ({
      theme: 'light',
      sidebarCollapsed: false,
      workflow: emptyWorkflow(),
      toggleTheme: () =>
        set((state) => ({ theme: state.theme === 'light' ? 'dark' : 'light' })),
      setTheme: (theme) => set({ theme }),
      toggleSidebar: () =>
        set((state) => ({ sidebarCollapsed: !state.sidebarCollapsed })),
      setWorkflowStep: (currentStep) =>
        set((state) => ({ workflow: { ...state.workflow, currentStep } })),
      setWorkflowContext: (context) =>
        set((state) => {
          // Treat a direct model_type update exactly like setModelType(). This
          // also protects callers that do not use the convenience action.
          if (hasOwn(context, 'modelType') && state.workflow.modelType !== null && context.modelType !== state.workflow.modelType) {
            const workflow = emptyWorkflow()
            workflow.modelType = context.modelType ?? null
            workflow.currentStep = workflow.modelType
              ? hasOwn(context, 'currentStep') ? context.currentStep as WorkflowStepId : 'upload'
              : 'model-type'
            return { workflow }
          }

          const workflow = { ...state.workflow, ...context }

          // An explicit ID is authoritative. When a caller only supplies a
          // resource object (the normal completion callback), derive its ID;
          // when an ID changes without a matching object, discard the old
          // cache rather than presenting it as the new resource.
          if (hasOwn(context, 'dataset')) {
            if (!hasOwn(context, 'datasetId')) workflow.datasetId = context.dataset?.id ?? null
            else if (context.dataset && context.dataset.id !== context.datasetId) workflow.dataset = null
          } else if (hasOwn(context, 'datasetId')
            && (context.datasetId === null || (workflow.dataset && workflow.dataset.id !== context.datasetId))) {
            workflow.dataset = null
          }

          if (hasOwn(context, 'preprocessTask') && !hasOwn(context, 'preprocessTaskId')) {
            workflow.preprocessTaskId = context.preprocessTask?.id ?? null
          } else if (hasOwn(context, 'preprocessTaskId')
            && (context.preprocessTaskId === null || (workflow.preprocessTask && workflow.preprocessTask.id !== context.preprocessTaskId))) {
            workflow.preprocessTask = null
          }

          if (hasOwn(context, 'split') && !hasOwn(context, 'splitId')) {
            workflow.splitId = context.split?.id ?? null
          } else if (hasOwn(context, 'splitId')
            && (context.splitId === null || (workflow.split && workflow.split.id !== context.splitId))) {
            workflow.split = null
          }

          if (hasOwn(context, 'trainingJob') && !hasOwn(context, 'trainingJobId')) {
            workflow.trainingJobId = context.trainingJob?.id ?? null
            // Older fixtures and responses sometimes omit this optional field.
            // Do not erase an explicit model ID in that case.
            if (!hasOwn(context, 'modelVersionId') && context.trainingJob
              && hasOwn(context.trainingJob, 'model_version_id')) {
              workflow.modelVersionId = context.trainingJob.model_version_id ?? null
            }
          } else if (hasOwn(context, 'trainingJobId')
            && (context.trainingJobId === null || (workflow.trainingJob && workflow.trainingJob.id !== context.trainingJobId))) {
            workflow.trainingJob = null
          }

          if (hasOwn(context, 'modelVersion') && !hasOwn(context, 'modelVersionId')) {
            workflow.modelVersionId = context.modelVersion?.id ?? null
          } else if (hasOwn(context, 'modelVersionId')
            && (context.modelVersionId === null || (workflow.modelVersion && workflow.modelVersion.id !== context.modelVersionId))) {
            workflow.modelVersion = null
          }

          return { workflow }
        }),
      setModelType: (modelType) =>
        set((state) => state.workflow.modelType === modelType
          ? { workflow: state.workflow }
          : { workflow: { ...emptyWorkflow(), modelType, currentStep: modelType ? 'upload' : 'model-type' } }),
      resetWorkflow: () => set({ workflow: emptyWorkflow() }),
    }),
    {
      name: 'model-training-platform-ui',
      partialize: (state) => ({
        theme: state.theme,
        sidebarCollapsed: state.sidebarCollapsed,
        workflow: persistedWorkflow(state.workflow),
      }),
      merge: (persisted, current) => {
        const saved = (persisted ?? {}) as Partial<AppState>
        const savedWorkflow = (saved.workflow ?? {}) as Partial<WorkflowDraft>

        // Migrate contexts written by earlier builds, where only the cache
        // object existed. Only infer an ID when the new ID property was absent;
        // an explicit persisted null must not be replaced by stale cache data.
        // Never merge those old objects into the live workflow: they are only
        // render caches, and showing one before the ID hydration completes can
        // make a deleted or replaced server resource look current.
        const workflow = emptyWorkflow()
        workflow.modelType = savedWorkflow.modelType ?? null
        workflow.datasetId = hasOwn(savedWorkflow, 'datasetId')
          ? savedWorkflow.datasetId ?? null
          : savedWorkflow.dataset?.id ?? null
        workflow.preprocessScriptId = savedWorkflow.preprocessScriptId ?? null
        workflow.preprocessTaskId = hasOwn(savedWorkflow, 'preprocessTaskId')
          ? savedWorkflow.preprocessTaskId ?? null
          : savedWorkflow.preprocessTask?.id ?? null
        workflow.splitId = hasOwn(savedWorkflow, 'splitId')
          ? savedWorkflow.splitId ?? null
          : savedWorkflow.split?.id ?? null
        workflow.trainScriptId = savedWorkflow.trainScriptId ?? null
        workflow.trainingJobId = hasOwn(savedWorkflow, 'trainingJobId')
          ? savedWorkflow.trainingJobId ?? null
          : savedWorkflow.trainingJob?.id ?? null
        workflow.modelVersionId = hasOwn(savedWorkflow, 'modelVersionId')
          ? savedWorkflow.modelVersionId ?? null
          : savedWorkflow.modelVersion?.id ?? null
        workflow.currentStep = savedWorkflow.currentStep ?? 'model-type'

        return {
          ...current,
          ...saved,
          workflow,
        }
      },
    },
  ),
)
