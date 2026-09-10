# Venturi Design & CFD

A parametric Venturi tube designer with its own CFD solver attached.

You give it an inlet diameter, a beta ratio and the two pressures you want.
It sizes the tube, builds the ISO 5167-4 geometry, meshes it, solves the
axisymmetric Navier-Stokes equations with a turbulence model, checks the
result against the discharge coefficient from the standard, and exports the
geometry and the flow field.

I wrote it because I wanted to understand projection methods by building one,
rather than by reading about one. There is no OpenFOAM or Fluent underneath:
the solver is roughly 900 lines of Python with Numba kernels.

## Running it

Python 3.10+. I've only tested it on 3.12 under Linux.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python main.py                     # interactive, asks you what you want
python main.py --D 0.1 --beta 0.5 --p-inlet 101325 --p-throat 95000
pytest tests/
```

First run takes an extra 30 seconds or so while Numba compiles. After that
it's cached.

`cadquery` isn't in requirements.txt on purpose. Install it if you want STEP
output; without it you still get the other nine files and a warning.

You end up with `.vts`/`.vtp` for ParaView, a `.dxf` of the profile, an
`.stl`, two `.svg` figures, a point cloud in `.ply` and `.csv`, and a
Markdown report.

## Speed

Single core, on my machine:

- 90x36 (the default): ~110k iterations, about a minute
- 128x51: ~272k iterations, about four and a half minutes

That iteration count is not a typo. See the last section.

## Why the solver is built this way

Most of these choices only make sense once you've seen the alternative fail,
so I've written down what went wrong rather than just what the code does.

**Finite volumes rather than finite differences.** The flux through a face is
one number, shared by the two cells on either side. Add up every cell's
balance and the interior terms cancel in pairs, leaving only `Q_in - Q_out`.
So mass conservation isn't something the solver works towards, it's just
algebra. Measured inlet/outlet imbalance is about 1e-12%, which is round-off
in the linear solve.

**The pressure matrix is assembled from the same face coefficients that
correct the fluxes.** A projection is exact only if the pressure matrix is
literally `D @ G`. Build the Laplacian separately with a slightly different
stencil and the projection stops removing divergence, but everything still
runs and converges to something plausible-looking. This is the failure mode I
started from, and it's nasty precisely because nothing crashes.

**Conical face fluxes use exact geometric coefficients** (`Cr`, `Cz`) instead
of `area * (u.n)` evaluated at the cell centre. With the naive version a
perfectly uniform flow generates a fake divergence of order h^2, and it does
it in the converging and diverging cones, exactly where you care. The
projection can't tell it's fake and invents pressure to cancel it. With the
split form, uniform flow gives zero divergence identically. Measured 2.8e-16.

**Fluxes are the primary variable**, cell velocities are reconstructed from
them. That's the MAC layout, so the pressure gradient correcting a flux only
touches the two adjacent cells. No checkerboard, no Rhie-Chow.

**Validation is against C_d, not ideal Bernoulli.** Bernoulli ignores
friction, so a correct result can't match it: real pressure drop is always a
few percent higher. "Within 5% of Bernoulli" fails good answers and passes
answers that are wrong in the convenient direction. C_d is measured
experimentally and tabulated in ISO 5167-4, which makes it an independent
reference. Taps are placed where the standard says: 0.5D upstream of the
convergent, and mid-throat.

## What I actually checked

| | |
|---|---|
| Mass balance in/out | ~1e-12 % |
| Divergence after projection | machine precision |
| Geometric conservation law | ~1e-16 |
| Order of convergence | 2.00 |
| dp/dz error, 80x40 Poiseuille | 0.031 % |
| Steady state vs time step | identical from CFL 0.8 to 0.1 |
| Tests | 20/20 |

The order and the dp/dz error come from laminar flow in a straight pipe,
where Hagen-Poiseuille gives the exact answer. It's the best check I have,
because it hits the viscous terms, the wall gradient and the axisymmetric
source all at once against something that can't be argued with.

The time-step check earned its place. An earlier version treated radial
diffusion implicitly and ran about 7x faster. It looked fine. Then I halved
the CFL and the error halved too, four times in a row. Error that scales with
dt isn't discretisation error, it means the converged solution itself depends
on the time step. The implicit preconditioner was acting on the acceleration
but not on the pressure gradient, which arrives later from the projection,
and the two don't commute. Taking it out moved the error from 1.51% to
0.031% and the order from 1.00 to 2.00.

Two things I have *not* shown:

- Grid independence on the Venturi itself. Only two grid levels. C_d goes
  0.9744 at 90x36 and 0.9792 at 128x51, right direction, but two points don't
  prove convergence.
- A clean install from requirements.txt in a fresh environment. I checked
  that every listed package is imported and nothing unlisted is needed, but
  that's reading the code, not installing it.

## Approximations

- Viscous term is `div(nu_eff * grad(u))`. The transpose term is dropped,
  which is the usual thing to do with an algebraic eddy viscosity.
- Turbulence is algebraic only: Baldwin-Lomax and mixing length with van
  Driest damping. No transport equation model. Asking for one raises
  NotImplementedError instead of quietly substituting something else.
- ISO 5167-4 defines C_d for 2e5 <= Re_D <= 2e6. The default case sits at
  Re_D ~ 9.2e4, under that. The comparison still runs but gets flagged as
  indicative. Raise the flow rate if you want it inside the range.

## The slow bit

Radial diffusion in the thin wall cells sets the time step. Those cells are
thin because you want y+ near 1, and eddy viscosity peaks there at roughly
200x molecular, so the diffusive limit ends up about 30x tighter than the
convective one. Hence six-figure iteration counts.

The implicit accelerator is still in the code:

```python
solve_fv(mesh, config, radial_implicit=True)   # ~7x fewer iterations
```

It's off by default for the reason above. Don't turn it on for anything you
care about.

The fix I know I need is a SIMPLE-style formulation with the Poisson matrix
weighted by 1/a_P, so momentum and pressure go through the same operator. I
tried an incremental version and it was unstable: residual dropped to 4.6e-4
then climbed back, and under-relaxing at 0.3 didn't save it. That's the next
job.

## Layout

```
main.py                CLI and interactive mode
venturi/
  config.py            inverse sizing, fluid presets
  geometry.py          ISO 5167-4 wall profile
  fvmesh.py            grid, face areas, GCL coefficients
  projection.py        divergence-free projection
  fvsolver.py          the solver
  turbulence.py        Baldwin-Lomax, mixing length
  validation_fv.py     C_d validation
  optimizer.py         shape optimisation
  export.py            output writers
  compat.py            adapter for the export module
tests/
```

## A note on AI

I used Claude as a coding assistant throughout: scheme design,
implementation, the diagnostic experiments that isolated the bugs above, and
the first draft of this README.

Everything in the table was measured by running the code, and the tests
reproduce it. The ISO values were looked up rather than recalled. The gaps
are listed above instead of being left out.

Judge it the way you'd judge any solver someone wrote themselves. The
conservation properties are provable and I measured them, the Poiseuille
check is against an exact solution, and the Venturi C_d comparison sits
outside the range the standard covers. If you spot something wrong, tell me.

## Licence

MIT, see LICENSE.
