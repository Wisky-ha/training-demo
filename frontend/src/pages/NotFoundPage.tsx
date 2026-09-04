import { Link } from 'react-router-dom'

export function NotFoundPage() {
  return (
    <div className="page-content placeholder-page">
      <div className="placeholder-inner">
        <div className="placeholder-icon" aria-hidden="true">?</div>
        <p className="eyebrow">404 / NOT FOUND</p>
        <h1 className="page-title">页面未找到</h1>
        <p className="page-description">当前地址没有对应的前端路由。</p>
        <Link className="back-link" to="/">返回平台概览</Link>
      </div>
    </div>
  )
}
