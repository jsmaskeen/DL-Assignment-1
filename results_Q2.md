# Q2: Optimizers on Rosenbrock - Results and Observations

## Parameters Used
- **Learning Rate**: 0.002
- **Start**: (-1.5, 1.5)
- **Target Minimum**: (1.0, 1.0)
- **Minimum Tolerance**: 1e-2

All optimizers were run for a maximum of 1000 steps. For Momentum, $\beta = 0.9$. For Adam, $\beta_1 = 0.9$, $\beta_2 = 0.999$. SGD and Minibatch-SGD used a noise scale of 0.15.

## Results Table

| Surface | Optimizer | Reached target | Steps | Final loss | Final x | Final y |
| --- | --- | --- | --- | --- | --- | --- |
| Rosenbrock | Gradient Descent | No | 1000 | 0.072170 | 0.731647 | 0.534055 |
| Rosenbrock | Momentum | Yes | 529 | 0.000020 | 0.995561 | 0.991124 |
| Rosenbrock | SGD | No | 1000 | 0.073427 | 0.729905 | 0.530580 |
| Rosenbrock | Minibatch-SGD | No | 1000 | 0.072334 | 0.731349 | 0.533605 |
| Rosenbrock | Adam | No | 1000 | 5.195003 | -1.278426 | 1.640519 |

## Observations and Explanations

### Why is Rosenbrock harder to optimize than Q1 surfaces?
The Rosenbrock function, $L(x,y) = (1-x)^2 + 100(y-x^2)^2$, is notoriously difficult to optimize. 
- **Non-convex Valley:** Unlike the Bowl or Ravine (which are convex), the Rosenbrock function has a deep, narrow, and curved (parabolic) valley. 
- **Ill-conditioning:** Similar to the Ravine, the gradients are very steep on the walls of the valley (due to the factor of 100) and extremely flat along the floor of the curved valley.
- **Changing Curvature:** As the optimizers travel along the valley floor to reach the global minimum at (1,1), the direction of the valley curves.

**Optimizer Performance:**
1. **Gradient Descent, SGD, and Minibatch-SGD**
   - **Performance**: They fail to reach the target within 1000 steps, getting stuck around $(x \approx 0.73, y \approx 0.53)$.
   - **Reason**: They are forced to use a very small learning rate (0.002) to avoid diverging on the steep walls initially. However, once they reach the flat valley floor, this small learning rate causes them to make agonizingly slow progress along the valley towards the minimum.
   - **Comparison to Q1**: This is similar to their failure in the Q1 Ravine, but compounded by the fact that the valley curves.

2. **Momentum**
   - **Performance**: The only optimizer to successfully reach the minimum (in 529 steps).
   - **Reason**: Momentum accumulates velocity along the valley floor, speeding up the optimization where gradients are very small, while dampening the oscillations across the steep walls.
   - **Comparison to Q1**: Just like it was the only optimizer to solve the Q1 Ravine, Momentum proves highly effective at navigating steep, narrow valleys.

3. **Adam**
   - **Performance**: Fails to reach the target within 1000 steps, ending up farther from the minimum than standard Gradient Descent.
   - **Reason**: On the Q1 Ravine, Adam was simply slow — its loss dropped over 99% and it kept moving steadily toward the minimum, just not fast enough to finish in time. Rosenbrock is different: Adam sets its step size using an average of *past* gradients, which works well when the steep and flat directions stay fixed (as in the Ravine), but Rosenbrock's valley curves, so the "steep" and "flat" directions keep rotating as x and y interact. Adam's step size lags behind this shift, so its steps end up pointed the wrong way rather than just being too small — causing it to overshoot and lose ground instead of slowly closing in.
   - **Comparison to Q1**: On the Ravine, Adam was heading the right way, just too slowly. On Rosenbrock, it actively moves the wrong way — showing that Adam struggles not with ill-conditioning itself, but with ill-conditioning that keeps changing direction.
