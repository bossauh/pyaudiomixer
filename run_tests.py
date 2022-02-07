import asyncio
from tests.output import TestOutput
from tests.input import TestInput
from tests.mixer import TestMixer


async def main() -> None:
    await TestInput().run()
    await TestOutput().run()
    await TestMixer().run()


if __name__ == "__main__":
    asyncio.run(main())
