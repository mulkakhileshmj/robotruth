# Guard detection on act_aloha_insertion: not run

Policy lerobot/act_aloha_sim_insertion_human scored 0% success in a 10-episode smoke test in gym_aloha/AlohaInsertion-v0, against roughly 30% on its model card. Calibrating a guard on a policy that does not perform the task would describe the nominal behaviour of a broken policy, so this combination is reported as unusable rather than measured. The loading path, not the guard, is the thing to fix.

COMBO_SKIPPED
