import { Navigate, Route, Routes } from 'react-router-dom'
import { AppLayout } from './components/layout/AppLayout'
import { HomePage } from './pages/HomePage'
import { NotFoundPage } from './pages/NotFoundPage'
import { PlaceholderPage } from './pages/PlaceholderPage'
import { WorkflowPage } from './pages/WorkflowPage'

function App() {
  return (
    <Routes>
      <Route element={<AppLayout />}>
        <Route index element={<HomePage />} />
        <Route path="workflow/:stepId?" element={<WorkflowPage />} />
        <Route
          path="models"
          element={
            <PlaceholderPage
              eyebrow="MODEL REGISTRY"
              title="模型版本"
              description="版本列表、评估结果与发布操作将在后续步骤接入。"
            />
          }
        />
        <Route
          path="mcp"
          element={
            <PlaceholderPage
              eyebrow="MCP SERVICE"
              title="服务说明"
              description="MCP 调用说明与接口联调内容将在后续步骤接入。"
            />
          }
        />
        <Route path="404" element={<NotFoundPage />} />
      </Route>
      <Route path="*" element={<Navigate to="/404" replace />} />
    </Routes>
  )
}

export default App
