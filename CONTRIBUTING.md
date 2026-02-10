# 贡献指南 (Contributing Guide)

感谢您对ERNIE5项目的关注！我们欢迎各种形式的贡献，包括bug报告、功能建议、文档改进和代码贡献。

## 行为准则

请在参与项目时保持尊重和建设性。我们致力于为所有人提供一个友好、安全和包容的环境。

## 如何贡献

### 报告Bug

如果您发现了bug，请创建一个issue并包含：

- 清晰的bug描述
- 重现步骤
- 预期行为
- 实际行为
- 环境信息（Python版本、操作系统等）
- 相关的错误日志或截图

### 建议新功能

我们欢迎新功能建议！请创建一个issue并说明：

- 功能的目的和用例
- 建议的实现方式
- 可能的替代方案
- 对现有功能的影响

### 提交代码

#### 1. Fork和克隆仓库

```bash
# Fork仓库到您的GitHub账户
# 然后克隆您的fork
git clone https://github.com/YOUR-USERNAME/ernie5.git
cd ernie5

# 添加上游仓库
git remote add upstream https://github.com/NJX-njx/ernie5.git
```

#### 2. 创建分支

从main分支创建一个新的功能分支：

```bash
git checkout main
git pull upstream main
git checkout -b feature/your-feature-name
```

分支命名约定：
- `feature/` - 新功能
- `fix/` - Bug修复
- `docs/` - 文档更新
- `refactor/` - 代码重构
- `test/` - 测试相关

#### 3. 开发环境设置

安装开发依赖：

```bash
# 安装基本依赖
pip install -r requirements.txt

# 安装开发工具
pip install pytest pytest-cov black isort flake8
```

#### 4. 进行更改

- 编写清晰、可维护的代码
- 遵循项目的代码风格
- 为新功能添加测试
- 更新相关文档

#### 5. 代码风格

我们使用以下工具来保持代码质量：

**格式化代码：**
```bash
# 使用Black格式化Python代码
black .

# 使用isort排序导入语句
isort .
```

**检查代码质量：**
```bash
# 使用Flake8检查代码
flake8 .
```

#### 6. 运行测试

在提交前，确保所有测试通过：

```bash
# 运行所有测试
pytest

# 运行测试并查看覆盖率
pytest --cov=ernie5 --cov-report=term
```

#### 7. 提交更改

编写清晰的提交消息：

```bash
git add .
git commit -m "类型: 简短描述

详细说明您的更改...

相关issue: #123"
```

提交消息类型：
- `feat:` - 新功能
- `fix:` - Bug修复
- `docs:` - 文档更新
- `style:` - 代码格式（不影响功能）
- `refactor:` - 代码重构
- `test:` - 添加或修改测试
- `chore:` - 构建过程或辅助工具的变动

#### 8. 推送更改

```bash
git push origin feature/your-feature-name
```

#### 9. 创建Pull Request

1. 在GitHub上访问您的fork
2. 点击 "New pull request"
3. 选择您的功能分支
4. 填写PR模板
5. 提交PR

#### 10. 代码审查

- 响应审查意见
- 根据反馈进行修改
- 推送新的提交以更新PR
- 所有对话解决后，PR将被合并

## 分支保护

main分支受到保护，这意味着：

- ✅ 不能直接推送到main
- ✅ 所有更改必须通过Pull Request
- ✅ 需要至少1个审批
- ✅ 必须通过所有CI检查
- ✅ 代码所有者必须审查相关更改

详细信息请参阅 [BRANCH_PROTECTION.md](BRANCH_PROTECTION.md)

## 代码审查指南

### 审查者

- 及时审查PR
- 提供建设性和具体的反馈
- 测试更改（如果需要）
- 批准或请求更改

### 提交者

- 清晰描述您的更改
- 保持PR小而专注
- 及时响应审查意见
- 耐心等待审查

## 开发提示

### 项目结构

```
ernie5/
├── configs/       # 模型、训练和Tokenizer配置
├── data/          # 数据集、采样器和Collator
├── models/        # 核心模型架构
├── tokenizers/    # 多模态Tokenizer
└── training/      # 训练循环、Loss、Scheduler
```

### 测试

- 为新功能编写单元测试
- 确保测试覆盖边缘情况
- 测试应该快速且独立
- 使用有意义的测试名称

### 文档

- 为公共API添加文档字符串
- 更新README（如果需要）
- 保持注释清晰和最新
- 使用中文或英文（保持一致）

## 获取帮助

如果您有任何问题：

1. 查看现有的issues和文档
2. 创建一个issue提问
3. 在PR中寻求反馈

## 许可

提交代码即表示您同意您的贡献将按照项目的许可证进行许可。

## 致谢

感谢所有为ERNIE5项目做出贡献的人！

---

**注意**: 首次贡献前，请阅读 [BRANCH_PROTECTION.md](BRANCH_PROTECTION.md) 了解分支保护规则。
