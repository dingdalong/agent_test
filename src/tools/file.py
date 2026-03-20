import os
from . import tool
from pydantic import BaseModel, Field

class WriteFileArgs(BaseModel):
    filename: str = Field(description="文件名，只能保存在 workspace 目录下")
    content: str = Field(description="文件内容")

@tool(model=WriteFileArgs, description="写入文件到工作区", sensitive=True)
async def write_file(filename: str, content: str) -> str:
    # 安全检查：确保路径在 workspace 内
    workspace = os.path.abspath("./workspace")
    os.makedirs(workspace, exist_ok=True)
    full_path = os.path.abspath(os.path.join(workspace, filename))
    if not full_path.startswith(workspace):
        return "错误：文件名不能包含路径遍历"
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"文件已保存：{filename}"
