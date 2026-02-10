# 分支保护快速参考 (Branch Protection Quick Reference)

## 🚀 快速开始

### 创建功能分支
```bash
git checkout main
git pull origin main
git checkout -b feature/your-feature
```

### 提交更改
```bash
# 格式化代码
black .
isort .

# 运行测试
pytest

# 提交
git add .
git commit -m "feat: 你的功能描述"
git push origin feature/your-feature
```

### 创建PR
1. 访问 GitHub 仓库
2. 点击 "New Pull Request"
3. 填写 PR 模板
4. 等待 CI 检查通过
5. 请求代码审查
6. 合并

## ✅ 检查清单

提交PR前：
- [ ] 代码已格式化 (`black .` 和 `isort .`)
- [ ] 所有测试通过 (`pytest`)
- [ ] 添加了新测试（如果需要）
- [ ] 更新了文档（如果需要）
- [ ] PR描述清晰完整

## 🔒 分支保护规则

- ✅ 不能直接推送到 main
- ✅ 需要至少 1 个审批
- ✅ 必须通过所有 CI 检查
- ✅ 代码所有者必须审查

## 🛠️ 开发工具

```bash
# 安装开发依赖
pip install black isort flake8 pytest pytest-cov

# 格式化
black .
isort .

# 检查代码质量
flake8 .

# 运行测试
pytest --cov=ernie5
```

## 📚 更多信息

- 详细指南: [BRANCH_PROTECTION.md](BRANCH_PROTECTION.md)
- 贡献指南: [CONTRIBUTING.md](CONTRIBUTING.md)
- 提交规范: 使用 `feat:`, `fix:`, `docs:` 等前缀

## ❓ 常见问题

**Q: CI 检查失败怎么办？**
A: 查看日志，修复问题，推送新提交。

**Q: 如何请求审查？**
A: 在 PR 页面右侧的 "Reviewers" 部分请求。

**Q: 合并策略是什么？**
A: 推荐使用 "Squash and merge"。

---
**提示**: 保持 PR 小而专注，响应审查意见要及时！
