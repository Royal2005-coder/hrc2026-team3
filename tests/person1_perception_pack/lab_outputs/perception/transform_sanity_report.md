# Transform Sanity Report

- Identity test: PASS
- det(R) = 1.000000 (should be ≈ 1.0): PASS

## Test points

| p_camera | p_base | in_workspace |
|----------|--------|-------------|
| [0.0, 0.0, 0.7] | [-0.4186, 0.032, 1.0967] | YES |
| [0.1, 0.0, 0.7] | [-0.4186, -0.068, 1.0967] | YES |
| [-0.1, 0.0, 0.7] | [-0.4186, 0.132, 1.0967] | YES |
| [0.0, 0.05, 0.5] | [-0.2254, 0.032, 1.0248] | YES |
