import numpy as np


class RandomJointActionGenerator:
    def __init__(self, num_joints: int, soft_limit: tuple[np.ndarray, np.ndarray]):
        self.num_joints = num_joints
        self.soft_limit = soft_limit

    @property
    def raw_action(self) -> np.ndarray:
        """Raw action in [-1, 1]."""
        return self._raw_action
    
    @property
    def processed_action(self) -> np.ndarray:
        """Processed action in joint limits."""
        return self._processed_action

    def unscale_transform(self, action: np.ndarray) -> np.ndarray:
        """Unscale action from [-1, 1] to joint limits."""
        lower_limit, upper_limit = self.soft_limit
        offset = (upper_limit + lower_limit) / 2
        return action * (upper_limit - lower_limit) / 2 + offset

    def sample(self) -> np.ndarray:
        # sample action in [-1, 1]
        self._raw_action = np.random.uniform(low=-1.0, high=1.0, size=(self.num_joints,))
        # unscale to joint limits
        self._processed_action = self.unscale_transform(self._raw_action)
        return self._raw_action