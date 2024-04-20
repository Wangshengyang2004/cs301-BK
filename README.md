# cs301-BK Project: Search Image by Image
## 需要完成的任务：
- 前端，streamlit开发， 画面简单不出bug即可，和后端通信，前端只负责把用户提交的图片推送到后端，并接收后端的图片并显示
- 后端，不采用torch-server/tensorflow-server，原生Fast API和Tensorflow，PyTorch开发

## 查看文档：
1. [Streamlit](https://docs.streamlit.io/develop/api-reference/widgets/st.link_button)
2. [FastAPI](https://fastapi.tiangolo.com/)
3. 知乎搜，都有的

## Github协作教程
当然可以！这里是一份简单易懂的教程，教您的团队成员如何使用GitHub提交代码给您。这份教程包括创建账号、配置环境、克隆仓库、提交更改和创建拉取请求（Pull Request, PR）的步骤。

### 第1步：创建并设置GitHub账号

1. **注册GitHub账号**  
   访问 [GitHub](https://github.com/) 并点击“Sign up”注册新账号。

2. **安装Git**  
   访问 [Git官网](https://git-scm.com/downloads) 下载适合您操作系统的版本，并按照安装向导完成安装。

3. **配置Git**  
   打开命令行或终端，设置您的用户名和电子邮箱：
   ```bash
   git config --global user.name "your_username"
   git config --global user.email "your_email@example.com"
   ```

### 第2步：克隆仓库

1. **克隆仓库**  
   打开命令行或终端，输入以下命令将仓库克隆到本地计算机：
   ```bash
   git clone https://github.com/Wangshengyang2004/cs301-BK.git
   ```

### 第3步：创建新的分支

创建一个新的分支以进行开发，保持`main`分支的稳定。

```bash
git checkout -b feature-branch # 你可以随意取名字，不要和别人的分支冲突
```
将`feature-branch`替换为您的分支名称，如`add-login-feature`。

### 第4步：进行更改并提交

1. **修改代码或添加新文件**  
   在本地编辑或添加文件。

2. **添加更改到暂存区**  
   完成更改后，将文件添加到Git暂存区：
   ```bash
   git add .
   ```
   或者添加指定文件：
   ```bash
   git add filename
   ```

3. **提交更改**  
   将暂存的更改提交到您的分支：
   ```bash
   git commit -m "Add a descriptive message here" #这里写的内容要具体写，好让团队成员都知道你做了什么更改
   ```

### 第5步：推送更改到GitHub

将您的更改推送到GitHub仓库：
```bash
git push origin feature-branch
```
确保替换`feature-branch`为您的实际分支名。

### 第6步：创建拉取请求（PR）

1. **访问GitHub仓库**  
   在浏览器中打开您的GitHub仓库。

2. **新建拉取请求**  
   点击“Pull requests” > “New pull request”。选择您的分支与`main`分支比较，然后点击“Create pull request”。

3. **描述并提交PR**  
   填写PR的标题和描述，说明您的更改，然后点击“Create pull request”。

### 第7步：审核和合并PR

作为项目维护者，您可以查看提交的PR，进行代码审查。如果一切顺利，可以将其合并到主分支。

希望这份教程对您有帮助！如果您有任何问题或需要进一步的解释，请随时告诉我。