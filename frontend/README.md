# 模型训练可视化平台前端

前端基础工程使用 **Vite + React + TypeScript**，路由使用 React Router，全局状态使用 Zustand。

## 开发启动

```bash
cd frontend
npm install
npm run dev
```

开发服务器默认地址为 `http://127.0.0.1:5173`。后端仍按 [`../backend/README.md`](../backend/README.md) 的方式单独启动（默认 `http://127.0.0.1:8000`）。

## 可用脚本

```bash
npm run dev        # 启动 Vite 开发服务器
npm run typecheck  # TypeScript 静态检查
npm run build      # 静态检查并构建生产资源
npm run preview    # 预览生产构建
```

## 当前范围

本步骤只初始化前端基础设施：应用布局、基础路由、浅色/深色主题和工作流上下文状态骨架。当前页面为导航和扩展占位，不连接后端接口，也未实现 CSV 上传、脚本管理、训练执行、评估、模型发布、版本管理或 MCP 业务能力；这些内容将在后续实施步骤中接入。

## 目录约定

```text
src/
├── components/layout/  # 应用级布局与导航
├── pages/              # 路由页面及后续业务页面入口
├── store/              # Zustand 全局状态
├── styles/             # 全局主题和布局样式
└── types/              # 工作流等共享类型
```
