"""Frozen, verified train demonstrations for the shared raw-action prompt.

These are complete offline examples from train tasks, not answers or expert
commands for the live task. Importing this module performs no file or runtime
access. The shared episode prompt uses these examples for Base, RFT and GRPO;
DEMONSTRATION_IDS must be excluded from training and evaluation task populations.

Provenance: runs_alfworld_tw_v2_recovery/proposed_demonstrations.json. Each
transcript preserves the real environment observations and executed actions.
"""
from __future__ import annotations

DEMONSTRATIONS_VERSION = "alfworld_train_demonstrations_v1"
PROVENANCE_ARTIFACT_SHA256 = "5501486dfd0e80f2ea4162ccaaf87d4dfd2ef9ac576b7f2cd2073ca0a2acfa50"

DEMONSTRATIONS = ({'id': 'alfworld:train/pick_and_place_simple-Book-None-Sofa-202/trial_T20190907_043640_929501',
  'source_split': 'train',
  'subject': 'pick_and_place_simple',
  'game_sha256': 'fd97532393059f2e6794ccc7d693822daf5cd3f77522dbec36eda91d317134a5',
  'trajectory_sha256': 'b69a0885700dc758762d955c71cbaddf9df6cc79a85228ed6fdef884f8b887a8',
  'transcript': 'Observation: -= Welcome to TextWorld, ALFRED! =-\n'
                '\n'
                'You are in the middle of a room. Looking quickly around you, you see a armchair 1, a '
                'coffeetable 1, a garbagecan 1, a sidetable 1, a sofa 1, and a tvstand 1.\n'
                '\n'
                'Your task is to: put a book in sofa.\n'
                'Action: go to sidetable 1\n'
                'Observation: You arrive at sidetable 1. On the sidetable 1, you see a book 3, a book 2, a '
                'book 1, a box 1, and a remotecontrol 1.\n'
                'Action: take book 3 from sidetable 1\n'
                'Observation: You pick up the book 3 from the sidetable 1.\n'
                'Action: go to sofa 1\n'
                'Observation: You arrive at sofa 1. On the sofa 1, you see a laptop 2, and a pillow 1.\n'
                'Action: move book 3 to sofa 1\n'
                'Observation: You move the book 3 to the sofa 1.'},
 {'id': 'alfworld:train/look_at_obj_in_light-CellPhone-None-DeskLamp-316/trial_T20190907_075509_267502',
  'source_split': 'train',
  'subject': 'look_at_obj_in_light',
  'game_sha256': '1deda6e05244204817de054d5c5b6ebdade1bfafda7ecd09158260bab27f606f',
  'trajectory_sha256': '893ff12724edf2c5d5760521d7387c33bada3cddf1bd65a073e5890c4c880034',
  'transcript': 'Observation: -= Welcome to TextWorld, ALFRED! =-\n'
                '\n'
                'You are in the middle of a room. Looking quickly around you, you see a bed 1, a desk 1, a '
                'drawer 2, a drawer 1, a garbagecan 1, a shelf 1, and a sidetable 1.\n'
                '\n'
                'Your task is to: look at cellphone under the desklamp.\n'
                'Action: go to desk 1\n'
                'Observation: You arrive at desk 1. On the desk 1, you see a alarmclock 1, a bowl 1, a cd 2, '
                'a cellphone 1, a creditcard 2, a creditcard 1, a desklamp 1, a houseplant 1, a keychain 1, '
                'a laptop 1, and a pen 1.\n'
                'Action: use desklamp 1\n'
                'Observation: You turn on the desklamp 1.\n'
                'Action: take cellphone 1 from desk 1\n'
                'Observation: You pick up the cellphone 1 from the desk 1.'},
 {'id': 'alfworld:train/pick_heat_then_place_in_recep-Potato-None-GarbageCan-14/trial_T20190909_114015_210188',
  'source_split': 'train',
  'subject': 'pick_heat_then_place_in_recep',
  'game_sha256': 'c76390c84bd20c0a4d563e0fe483407b644a7df404b4a02b99bbae439d4344fd',
  'trajectory_sha256': '1e4cf9c66b0d31b5d212a423d1dc0100a1ed5c8e77d8454cd90cd7b4f5a9db5a',
  'transcript': 'Observation: -= Welcome to TextWorld, ALFRED! =-\n'
                '\n'
                'You are in the middle of a room. Looking quickly around you, you see a cabinet 8, a cabinet '
                '7, a cabinet 6, a cabinet 5, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a '
                'coffeemachine 1, a countertop 2, a countertop 1, a fridge 1, a garbagecan 1, a microwave 1, '
                'a sinkbasin 1, a stoveburner 4, a stoveburner 3, a stoveburner 2, a stoveburner 1, and a '
                'toaster 1.\n'
                '\n'
                'Your task is to: heat some potato and put it in garbagecan.\n'
                'Action: go to countertop 1\n'
                'Observation: You arrive at countertop 1. On the countertop 1, you see a butterknife 3, a '
                'butterknife 2, a butterknife 1, a cup 2, a egg 1, a fork 2, a knife 2, a knife 1, a lettuce '
                '3, a lettuce 2, a lettuce 1, a pan 1, a potato 2, a potato 1, a saltshaker 3, a saltshaker '
                '1, and a tomato 3.\n'
                'Action: take potato 2 from countertop 1\n'
                'Observation: You pick up the potato 2 from the countertop 1.\n'
                'Action: go to microwave 1\n'
                'Observation: You arrive at microwave 1. The microwave 1 is closed.\n'
                'Action: heat potato 2 with microwave 1\n'
                'Observation: You heat the potato 2 using the microwave 1.\n'
                'Action: go to garbagecan 1\n'
                'Observation: You arrive at garbagecan 1. On the garbagecan 1, you see a dishsponge 2, and a '
                'tomato 1.\n'
                'Action: move potato 2 to garbagecan 1\n'
                'Observation: You move the potato 2 to the garbagecan 1.'})

DEMONSTRATION_IDS = frozenset(demonstration["id"] for demonstration in DEMONSTRATIONS)
DEMONSTRATION_SOURCE_IDS = frozenset(task_id.removeprefix("alfworld:") for task_id in DEMONSTRATION_IDS)


def render_demonstration_prefix() -> str:
    """Render only the examples and new-task separator, without base instructions."""
    examples = [
        f"Example {index}:\n{demonstration['transcript']}\nTask completed."
        for index, demonstration in enumerate(DEMONSTRATIONS, start=1)
    ]
    return (
        "The following examples are complete independent tasks. Follow the same command format.\n\n"
        + "\n\n".join(examples)
        + "\n\nNow complete a new task using its observations."
    )
