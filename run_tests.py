import asyncio
from tests.output import TestOutput


async def main() -> None:
    await TestOutput().run()


if __name__ == "__main__":
    asyncio.run(main())
