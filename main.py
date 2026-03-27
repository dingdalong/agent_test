# main.py
"""Agent 入口。"""

import asyncio

from src.app import AgentApp
from src.interfaces.cli import CLIInterface


async def main():
    app = AgentApp(ui=CLIInterface())
    await app.setup()
    try:
        await app.run()
    finally:
        await app.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
