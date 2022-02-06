import asyncio
from tests.output import TestOutput
from tests.input import TestInput


async def main() -> None:
    await TestInput().run()
    await TestOutput().run()


if __name__ == "__main__":
    asyncio.run(main())
