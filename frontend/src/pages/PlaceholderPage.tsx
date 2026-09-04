import { Link } from 'react-router-dom'

interface PlaceholderPageProps {
  eyebrow: string
  title: string
  description: string
}

export function PlaceholderPage({ eyebrow, title, description }: PlaceholderPageProps) {
  return (
    <div className="page-content placeholder-page">
      <div className="placeholder-inner">
        <div className="placeholder-icon" aria-hidden="true">⌘</div>
        <p className="eyebrow">{eyebrow}</p>
        <h1 className="page-title">{title}</h1>
        <p className="page-description">{description}</p>
        <Link className="back-link" to="/">返回平台概览</Link>
      </div>
    </div>
  )
}
