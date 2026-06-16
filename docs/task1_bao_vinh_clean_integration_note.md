# Task 1 - Bao/Vinh Clean Integration Note

This note documents how Bao and Vinh's latest pushed branches were handled before teacher review.

## Why not merge the original branches directly?

### Bao branch: `bao/planner_support`
The branch contains useful GA/RL/Pinochio/IK and motion-support code, but also includes large/generated artifacts such as:

- `Part_Sorting/`
- videos (`*.mp4`)
- parquet dataset files (`*.parquet`)
- `.DS_Store`
- conflicts in `.gitignore` and `src/task1/motion.py`

Therefore, the branch was not merged directly into `staging`.

### Vinh branch: `vinh/motion`
The branch changes repository structure heavily by moving baseline/config/source files into `Ubtech_sim/` and deleting/moving several existing core files/scripts.

Therefore, the branch was not merged directly into root staging because it could break the current reviewable structure.

## Clean integration policy

To preserve reviewability:

- Bao's safe code/docs were copied under `contrib/bao_planner_support/`.
- Bao's conflicting `motion.py` was saved as `contrib/bao_planner_support/src/task1/motion_bao_experimental.py`.
- Vinh's moved `Ubtech_sim` work was copied under `contrib/vinh_motion/`.
- Dataset/video/model artifacts were excluded.
- Root staging structure was kept stable for teacher review.

## PM note

The original Bao/Vinh branches are recorded in Git history with artifact-safe merge commits, but destructive/generated files are intentionally excluded from the final staging tree.
