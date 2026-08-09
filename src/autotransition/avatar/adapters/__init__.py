"""External model adapters used by the avatar pipeline."""

from autotransition.avatar.adapters.image_generator import CommandImageGenerator
from autotransition.avatar.adapters.mesh_generator import CommandMeshGenerator
from autotransition.avatar.adapters.rig_generator import CommandReskinGenerator, CommandRigGenerator

__all__ = ["CommandImageGenerator", "CommandMeshGenerator", "CommandReskinGenerator", "CommandRigGenerator"]
