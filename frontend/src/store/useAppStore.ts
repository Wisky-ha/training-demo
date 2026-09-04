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
  resetWorkflow: () => void
}

const initialWorkflow: WorkflowDraft = {
  modelType: null,
  datasetId: null,
  dataset: null,
  preprocessScriptId: null,
  preprocessTaskId: null,
  preprocessTask: null,
  split: null,
  trainScriptId: null,
  trainingJobId: null,
  trainingJob: null,
  evaluation: null,
  modelVersion: null,
  currentStep: 'model-type',
}

export const useAppStore = create<AppState>()(
  persist(
    (set) => ({
      theme: 'light',
      sidebarCollapsed: false,
      workflow: initialWorkflow,
      toggleTheme: () =>
        set((state) => ({ theme: state.theme === 'light' ? 'dark' : 'light' })),
      setTheme: (theme) => set({ theme }),
      toggleSidebar: () =>
        set((state) => ({ sidebarCollapsed: !state.sidebarCollapsed })),
      setWorkflowStep: (currentStep) =>
        set((state) => ({ workflow: { ...state.workflow, currentStep } })),
      setWorkflowContext: (context) =>
        set((state) => ({ workflow: { ...state.workflow, ...context } })),
      resetWorkflow: () => set({ workflow: initialWorkflow }),
    }),
    {
      name: 'model-training-platform-ui',
      partialize: (state) => ({
        theme: state.theme,
        sidebarCollapsed: state.sidebarCollapsed,
        workflow: state.workflow,
      }),
    },
  ),
)
