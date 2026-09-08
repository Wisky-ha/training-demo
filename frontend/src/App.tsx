import { Navigate, Route, Routes } from 'react-router-dom'
import { AppLayout } from './components/layout/AppLayout'
import { HomePage } from './pages/HomePage'
import { NotFoundPage } from './pages/NotFoundPage'
import { McpPage } from './pages/McpPage'
import { ModelVersionsPage } from './pages/ModelVersionsPage'
import { WorkflowPage } from './pages/WorkflowPage'
import { AuditPage } from './pages/AuditPage'

function App() {
  return (
    <Routes>
      <Route element={<AppLayout />}>
        <Route index element={<HomePage />} />
        <Route path="workflow/:stepId?" element={<WorkflowPage />} />
        <Route path="models" element={<ModelVersionsPage />} />
        <Route path="audit" element={<AuditPage />} />
        <Route path="mcp" element={<McpPage />} />
        <Route path="404" element={<NotFoundPage />} />
      </Route>
      <Route path="*" element={<Navigate to="/404" replace />} />
    </Routes>
  )
}

export default App
