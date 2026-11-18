import asyncio

from viam.module.module import Module

import sdk_reconfigure
from models.sam3_segments import Sam3Segments  # noqa: F401  registers the model


if __name__ == "__main__":
    sdk_reconfigure.install()
    asyncio.run(Module.run_from_registry())
