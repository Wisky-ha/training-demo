import { Link } from 'react-router-dom'

const foundationItems = [
  {
    index: '01 / ROUTING',
    title: '应用路由骨架',
    description: '已建立概览、训练工作流、模型版本和服务说明的扩展入口。',
  },
  {
    index: '02 / STATE',
    title: '全局状态管理',
    description: '主题、侧栏和训练任务上下文已集中管理，后续流程可以逐步接入。',
  },
  {
    index: '03 / THEME',
    title: '统一视觉基础',
    description: '提供响应式布局、浅色/深色主题和可复用的界面设计变量。',
  },
]

export function HomePage() {
  return (
    <div className="page-content">
      <section className="hero">
        <div className="hero-copy">
          <div className="hero-kicker"><span /> FRONTEND FOUNDATION</div>
          <p className="eyebrow">MODEL TRAINING VISUALIZATION PLATFORM</p>
          <h1 className="page-title">让训练流程，清晰可见。</h1>
          <p className="page-description">
            前端基础工程已完成初始化。这里是面向后续训练工作流的应用入口，当前仅提供布局、导航、主题和状态管理骨架。
          </p>
          <Link className="workflow-link" to="/workflow">查看工作流骨架 <span aria-hidden="true">→</span></Link>
        </div>
      </section>

      <div className="section-heading">
        <div>
          <h2>基础能力</h2>
          <p>Foundation layer</p>
        </div>
      </div>
      <section className="foundation-grid" aria-label="前端基础能力">
        {foundationItems.map((item) => (
          <article className="foundation-card" key={item.index}>
            <div className="card-index">{item.index}</div>
            <h3>{item.title}</h3>
            <p>{item.description}</p>
          </article>
        ))}
      </section>
    </div>
  )
}
