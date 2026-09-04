import { useAppStore } from '../store/useAppStore'
import { workflowSteps } from '../types/workflow'

export function WorkflowPage() {
  const currentStep = useAppStore((state) => state.workflow.currentStep)
  const activeIndex = workflowSteps.findIndex((step) => step.id === currentStep)

  return (
    <div className="page-content">
      <div className="workflow-header">
        <div>
          <p className="eyebrow">TRAINING WORKFLOW / FOUNDATION</p>
          <h1 className="page-title">训练工作流</h1>
          <p className="page-description">
            流程导航已预留。各业务步骤将在后续实施步骤中逐一接入，此处暂不调用后端接口。
          </p>
        </div>
      </div>

      <section className="step-shell" aria-label="训练流程步骤">
        <div className="step-list">
          {workflowSteps.map((step, index) => (
            <div className={`step-item${index === activeIndex ? ' active' : ''}`} key={step.id}>
              <div className="step-number">{String(index + 1).padStart(2, '0')}</div>
              <span className="step-label">{step.label}</span>
              <span className="step-caption">{step.caption}</span>
            </div>
          ))}
        </div>

        <div className="workflow-placeholder">
          <div>
            <div className="placeholder-icon" aria-hidden="true">✦</div>
            <h2>步骤内容预留</h2>
            <p>当前步骤状态由全局工作流上下文承载。数据上传、脚本选择、训练执行和评估等业务能力尚未在本步骤实现。</p>
            <div className="scope-note"><span aria-hidden="true">●</span> 本步骤范围：前端基础设施</div>
          </div>
        </div>
      </section>
    </div>
  )
}
