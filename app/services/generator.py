from abc import ABC, abstractmethod
from typing import List, Any
import numpy as np


class Generator(ABC):
    """
    Abstract interface for content generators (images or video frames).
    """
    @abstractmethod
    async def generate(self, prompt: List[str] | str, **kwargs) -> List[np.ndarray]:
        """
        Generate content (images or video frames) based on text prompts.

        Args:
            prompt: List of text prompts to guide generation.
            **kwargs: Generator-specific parameters.

        Returns:
            List of numpy arrays representing generated content.
        """
        pass