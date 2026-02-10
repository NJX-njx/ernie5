# Main分支保护指南 (Main Branch Protection Guide)

本文档说明如何为ERNIE5项目的main分支配置保护规则，以确保代码质量和项目稳定性。

## 为什么需要分支保护？

分支保护可以：
- 防止直接推送到main分支，避免未经审查的代码进入主分支
- 要求代码审查，确保代码质量
- 强制执行自动化测试，防止破坏性更改
- 维护清晰的git历史记录
- 促进团队协作和知识共享

## GitHub分支保护规则配置

作为仓库管理员，请按照以下步骤在GitHub上配置分支保护：

### 1. 访问分支保护设置

1. 进入GitHub仓库页面
2. 点击 **Settings** (设置)
3. 在左侧菜单中选择 **Branches** (分支)
4. 在 "Branch protection rules" 部分，点击 **Add rule** (添加规则)

### 2. 配置保护规则

在 "Branch name pattern" 中输入 `main`，然后启用以下规则：

#### 必需的保护规则：

**✅ Require a pull request before merging (合并前需要拉取请求)**
- 启用此选项以防止直接推送到main分支
- 配置选项：
  - ✅ **Require approvals**: 至少需要 **1** 个审批
  - ✅ **Dismiss stale pull request approvals when new commits are pushed**: 当有新提交时，撤销过时的审批
  - ✅ **Require review from Code Owners**: 需要代码所有者审查（基于 `.github/CODEOWNERS` 文件）

**✅ Require status checks to pass before merging (合并前需要通过状态检查)**
- 启用此选项以要求CI/CD检查通过
- ✅ **Require branches to be up to date before merging**: 要求分支在合并前是最新的
- 添加必需的状态检查：
  - `test (3.8)` - Python 3.8测试
  - `test (3.9)` - Python 3.9测试
  - `test (3.10)` - Python 3.10测试
  - `test (3.11)` - Python 3.11测试
  - `lint` - 代码质量检查

**✅ Require conversation resolution before merging (合并前需要解决所有对话)**
- 确保所有PR评论都已解决

**✅ Require signed commits (需要签名提交)**
- 可选但推荐，增强安全性

**✅ Require linear history (需要线性历史)**
- 可选，防止合并提交，保持清晰的git历史

**✅ Include administrators (包括管理员)**
- 推荐启用，使规则适用于所有人，包括管理员

#### 推荐的附加规则：

**✅ Restrict who can push to matching branches**
- 限制只有特定用户或团队可以推送（即使通过PR）

**✅ Allow force pushes - Specify who can force push**
- 默认禁用强制推送
- 如需允许，仅限特定维护者

**✅ Allow deletions**
- 默认禁用分支删除

### 3. 保存规则

点击 **Create** 或 **Save changes** 保存分支保护规则。

## 使用分支保护的工作流程

### 开发者工作流程：

1. **从main创建功能分支**
   ```bash
   git checkout main
   git pull origin main
   git checkout -b feature/your-feature-name
   ```

2. **进行开发并提交更改**
   ```bash
   git add .
   git commit -m "描述你的更改"
   git push origin feature/your-feature-name
   ```

3. **创建Pull Request**
   - 在GitHub上创建PR
   - 填写PR模板（使用 `.github/pull_request_template.md`）
   - 等待CI检查通过
   - 请求代码审查

4. **响应审查意见**
   - 根据审查意见修改代码
   - 推送新的提交
   - 确保所有对话都已解决

5. **合并PR**
   - 所有检查通过后
   - 获得必需的审批后
   - 点击 "Merge pull request"

### 审查者工作流程：

1. 审查代码更改
2. 运行代码进行测试（如果需要）
3. 提供建设性反馈
4. 批准PR或请求更改

## 自动化检查

项目配置了以下自动化检查（参见 `.github/workflows/ci.yml`）：

### 测试 (Test)
- 在多个Python版本（3.8, 3.9, 3.10, 3.11）上运行测试套件
- 生成代码覆盖率报告
- 必须全部通过才能合并

### 代码质量 (Lint)
- **Black**: 代码格式检查
- **isort**: 导入语句排序检查
- **Flake8**: 代码质量和风格检查
- 语法错误会导致检查失败

## 代码所有者 (CODEOWNERS)

`.github/CODEOWNERS` 文件定义了代码所有权：

- 核心模型代码 (`/ernie5/models/`) 需要特定维护者审查
- 配置更改 (`/ernie5/configs/`) 需要审查
- 训练代码 (`/ernie5/training/`) 需要审查
- CI/CD更改 (`/.github/`) 需要审查

## 紧急情况处理

在紧急情况下（如生产环境严重bug）：

1. **仍然使用PR流程** - 即使在紧急情况下也应创建PR
2. **加急审查** - 通知审查者加急处理
3. **后续跟进** - 紧急修复后，后续进行适当的测试和文档更新

## 常见问题

### Q: 我可以直接推送到main分支吗？
A: 不可以。启用分支保护后，所有更改都必须通过Pull Request。

### Q: 如果CI检查失败怎么办？
A: 查看失败的日志，修复问题，然后推送新的提交。CI会自动重新运行。

### Q: 我需要多少个审批？
A: 根据配置，通常需要至少1个审批。代码所有者的审批可能是必需的。

### Q: 如何请求审查？
A: 在PR页面，使用右侧的 "Reviewers" 部分请求特定人员审查。

### Q: 合并策略是什么？
A: 推荐使用 "Squash and merge" 以保持清晰的提交历史。

## 最佳实践

1. **保持PR小而专注** - 小的PR更容易审查和合并
2. **编写清晰的PR描述** - 使用提供的模板
3. **及时响应审查意见** - 加快合并速度
4. **运行本地测试** - 在推送前本地运行 `pytest`
5. **遵循代码风格** - 使用 `black` 和 `isort` 格式化代码
6. **编写测试** - 为新功能和bug修复添加测试
7. **更新文档** - 相应地更新README和其他文档

## 相关文件

- `.github/CODEOWNERS` - 代码所有权定义
- `.github/workflows/ci.yml` - CI/CD配置
- `.github/pull_request_template.md` - PR模板
- `CONTRIBUTING.md` - 贡献指南

## 参考资源

- [GitHub Branch Protection Documentation](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches)
- [GitHub CODEOWNERS Documentation](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
