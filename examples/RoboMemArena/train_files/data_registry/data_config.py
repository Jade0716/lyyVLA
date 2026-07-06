"""RoboMemArena data config and dataset mixture registration."""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)


class RoboMemArenaFrankaDataConfig:
    """Franka delta-EEF controls produced by the RoboMemArena converter."""

    embodiment_tag = EmbodimentTag.FRANKA
    video_keys = [
        "video.primary_image",
        "video.wrist_image",
    ]
    state_keys = [
        "state.x",
        "state.y",
        "state.z",
        "state.roll",
        "state.pitch",
        "state.yaw",
        "state.pad",
        "state.gripper",
    ]
    action_keys = [
        "action.x",
        "action.y",
        "action.z",
        "action.roll",
        "action.pitch",
        "action.yaw",
        "action.gripper",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    state_indices = [0]
    action_indices = list(range(8))

    def modality_config(self):
        return {
            "video": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.video_keys,
            ),
            "state": ModalityConfig(
                delta_indices=self.state_indices,
                modality_keys=self.state_keys,
            ),
            "action": ModalityConfig(
                delta_indices=self.action_indices,
                modality_keys=self.action_keys,
            ),
            "language": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.language_keys,
            ),
        }

    def transform(self):
        state_normalization = {key: "q99" for key in self.state_keys}
        action_normalization = {key: "clip" for key in self.action_keys}
        return ComposedModalityTransform(
            transforms=[
                StateActionToTensor(apply_to=self.state_keys),
                StateActionTransform(
                    apply_to=self.state_keys,
                    normalization_modes=state_normalization,
                ),
                StateActionToTensor(apply_to=self.action_keys),
                StateActionTransform(
                    apply_to=self.action_keys,
                    normalization_modes=action_normalization,
                ),
            ]
        )


ROBOT_TYPE_CONFIG_MAP = {
    "robomemarena_franka": RoboMemArenaFrankaDataConfig(),
}


DATASET_NAMED_MIXTURES = {
    # Expected path:
    #   <datasets.vla_data.data_root_dir>/<dataset_name>/meta/info.json
    "robomemarena_all": [
        ("robomemarena_lerobot", 1.0, "robomemarena_franka"),
    ],
    "robomemarena_counting": [
        ("robomemarena_lerobot/Multi-Object_Counting", 1.0, "robomemarena_franka"),
    ],
    "robomemarena_sequence": [
        ("robomemarena_lerobot/Multi-Object_Sequence", 1.0, "robomemarena_franka"),
    ],
    "robomemarena_transferring": [
        ("robomemarena_lerobot/Multi-Object_Transferring", 1.0, "robomemarena_franka"),
    ],
    "robomemarena_occlusion": [
        ("robomemarena_lerobot/Multi-Object_Occlusion", 1.0, "robomemarena_franka"),
    ],
}
