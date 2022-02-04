import asyncio
from tests.track import TestTrack


async def main() -> None:
    await TestTrack().run()


if __name__ == "__main__":
    asyncio.run(main())
