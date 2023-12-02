from gym import ObservationWrapper


class TransposeObservation(ObservationWrapper):
    r"""Tranpose the pixel observation shape from (H, W, C) to (C, H, W) for pytorch CNN input."""

    def observation(self, observation):
        observation["pixels"] = observation["pixels"].transpose(2, 0, 1)
        return observation
