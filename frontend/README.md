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

当前页面仍是导航和扩展占位，不执行业务 API 调用。已建立前后端共享的数据契约和统一 API 客户端，供后续步骤接入；CSV 上传、脚本管理、训练执行、评估、模型发布、版本管理和 MCP 页面业务仍未实现。

API 客户端入口为 `src/api/index.ts`，默认请求同源 `/api`。设置 `VITE_API_BASE_URL` 后可指定后端 origin（例如 `http://127.0.0.1:8000`，客户端会补齐 `/api`）；具体契约集中在 `src/types/contracts.ts`。

## 目录约定

```text
src/
├── api/                # 统一 API 客户端与请求入口
├── components/layout/  # 应用级布局与导航
├── pages/              # 路由页面及后续业务页面入口
├── store/              # Zustand 全局状态
├── styles/             # 全局主题和布局样式
└── types/              # 工作流与前后端数据契约
```
