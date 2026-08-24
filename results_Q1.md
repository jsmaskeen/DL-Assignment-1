# Q1: Optimizers on Loss Surfaces - Results and Observations

## Parameters Used
- **Bowl**: Learning Rate = 0.08, Start = (-4, 4), Target = (0, 0)
- **Ravine**: Learning Rate = 0.005, Start = (-4, 3), Target = (0, 0)
- **Saddle**: Learning Rate = 0.05, Start = (-1.5, 0.001), Escape Threshold = 0.25 (target reached when $|y| \ge 0.25$)

All optimizers were run for a maximum of 1000 steps. For Momentum, $\beta = 0.9$. For Adam, $\beta_1 = 0.9$, $\beta_2 = 0.999$. SGD and Minibatch-SGD used a noise scale of 0.15.

## Results Table

| Surface | Optimizer | Reached target | Steps | Final loss | Final x | Final y |
| --- | --- | --- | --- | --- | --- | --- |
| Bowl | Gradient Descent | Yes | 50 | 0.000001 | -0.000655 | 0.000655 |
| Bowl | Momentum | Yes | 111 | 0.000001 | 0.000612 | -0.000612 |
| Bowl | SGD | Yes | 848 | 0.000000 | -0.000039 | -0.000052 |
| Bowl | Minibatch-SGD | Yes | 203 | 0.000001 | -0.000484 | 0.000699 |
| Bowl | Adam | Yes | 144 | 0.000000 | 0.000496 | -0.000496 |
| Ravine | Gradient Descent | No | 1000 | 1800.000000 | -0.000173 | 3.000000 |
| Ravine | Momentum | Yes | 130 | 0.000181 | 0.000147 | -0.000950 |
| Ravine | SGD | No | 1000 | 1780.834351 | -0.002060 | 2.983986 |
| Ravine | Minibatch-SGD | No | 1000 | 1794.060181 | 0.000133 | 2.995046 |
| Ravine | Adam | No | 1000 | 7.742406 | -0.642013 | 0.191445 |
| Saddle | Gradient Descent | Yes | 58 | -0.063310 | -0.003328 | 0.251638 |
| Saddle | Momentum | Yes | 22 | 0.031467 | -0.329525 | 0.277705 |
| Saddle | SGD | Yes | 45 | -0.064132 | -0.002524 | -0.253255 |
| Saddle | Minibatch-SGD | Yes | 49 | -0.071949 | -0.012083 | 0.268504 |
| Saddle | Adam | Yes | 6 | 1.372177 | -1.201532 | 0.267398 |

## Observations and Explanations

### Bowl (Simple Convex)
- All optimizers easily find the minimum. 
- **Gradient Descent** converges extremely fast in just 50 steps because the surface has uniform curvature ($x^2 + y^2$).
- **SGD** and **Minibatch-SGD** take longer due to the added noise which causes them to bounce around the minimum before reaching the target tolerance.

### Ravine (Ill-conditioned)
- Only **Momentum** succeeds in reaching the minimum within 1000 steps (130 steps).
- The ravine has very steep walls along the $y$-axis (due to $200y^2$) and a very flat bottom along the $x$-axis. Gradient Descent, SGD, and Minibatch-SGD bounce back and forth along the $y$-axis walls while making very little progress along $x$.
- Momentum dampens the oscillations across the walls and accelerates progress along the flat bottom, efficiently solving the ill-conditioned curvature.
- Adam struggles here possibly due to its adaptive scaling making the learning rate too small along the slow axis, leading to no convergence within 1000 steps.

### Saddle (Escaping Saddle Point)
- The assignment asks to report how many steps it took to move clearly away from the center (defined as $|y| \ge 0.25$).
- **Adam** escapes the saddle extremely quickly (just 6 steps) because its adaptive learning rate scales up the very small initial gradient along the escape direction ($y$-axis).
- **Momentum** also escapes quickly (22 steps) by accumulating velocity.
- The standard **Gradient Descent** and **SGD/Minibatch-SGD** variants take slightly longer (around 45-60 steps) but still successfully escape because the starting point (-1.5, 0.001) is slightly off-center along $y$, allowing the gradient in that direction to eventually grow.
