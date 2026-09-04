import { useEffect } from 'react'
import { NavLink, Outlet, useLocation } from 'react-router-dom'
import { useAppStore } from '../../store/useAppStore'

const navigation = [
  { to: '/', label: '平台概览', icon: '⌂', end: true },
  { to: '/workflow', label: '训练工作流', icon: '◈' },
  { to: '/models', label: '模型版本', icon: '◫' },
  { to: '/mcp', label: '服务说明', icon: '⌁' },
]

const pageNames: Record<string, string> = {
  '/': '平台概览',
  '/workflow': '训练工作流',
  '/models': '模型版本',
  '/mcp': '服务说明',
  '/404': '页面未找到',
}

export function AppLayout() {
  const location = useLocation()
  const theme = useAppStore((state) => state.theme)
  const sidebarCollapsed = useAppStore((state) => state.sidebarCollapsed)
  const toggleTheme = useAppStore((state) => state.toggleTheme)
  const toggleSidebar = useAppStore((state) => state.toggleSidebar)

  useEffect(() => {
    document.documentElement.dataset.theme = theme
  }, [theme])

  const pageName = pageNames[location.pathname] ?? '平台'

  return (
    <div className={`app-shell${sidebarCollapsed ? ' sidebar-collapsed' : ''}`}>
      <aside className="sidebar">
        <div className="brand" aria-label="模型训练可视化平台">
          <div className="brand-mark">MT</div>
          <div>
            <div className="brand-name">模型训练平台</div>
            <div className="brand-caption">MODEL STUDIO</div>
          </div>
        </div>

        <div className="sidebar-label">Workspace</div>
        <nav className="nav-list" aria-label="主导航">
          {navigation.map((item) => (
            <NavLink
              className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}
              end={item.end}
              key={item.to}
              to={item.to}
            >
              <span aria-hidden="true" className="nav-icon">{item.icon}</span>
              <span>{item.label}</span>
            </NavLink>
          ))}
        </nav>

        <div className="sidebar-footer">
          <div className="workspace-name"><span className="workspace-dot" />内部演示工作区</div>
          <div className="workspace-caption">FOUNDATION v0.1</div>
        </div>
      </aside>

      <div className="main-shell">
        <header className="topbar">
          <div className="breadcrumb">
            <strong>工作台</strong>
            <span className="breadcrumb-separator">/</span>
            <span>{pageName}</span>
          </div>
          <div className="topbar-actions">
            <div className="foundation-status"><span className="status-pulse" />前端基础已就绪</div>
            <button
              aria-label={sidebarCollapsed ? '展开侧栏' : '收起侧栏'}
              className="icon-button"
              onClick={toggleSidebar}
              type="button"
            >
              {sidebarCollapsed ? '»' : '«'}
            </button>
            <button
              aria-label={theme === 'light' ? '切换为深色主题' : '切换为浅色主题'}
              className="icon-button"
              onClick={toggleTheme}
              type="button"
            >
              {theme === 'light' ? '☾' : '☀'}
            </button>
          </div>
        </header>
        <main><Outlet /></main>
      </div>
    </div>
  )
}
