# Transform Sanity Report

- Identity test: PASS
- det(R) = 1.000000 (should be ≈ 1.0): PASS

## Test points

| p_camera | p_base | in_workspace |
|----------|--------|-------------|
| [0.0, 0.0, 0.7] | [0.736, 0.032, 0.3049] | YES |
| [0.1, 0.0, 0.7] | [0.736, -0.068, 0.3049] | YES |
| [-0.1, 0.0, 0.7] | [0.736, 0.132, 0.3049] | YES |
| [0.0, 0.05, 0.5] | [0.5428, 0.032, 0.3768] | YES |
