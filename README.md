# Venturi Design & CFD

Give it an inlet diameter, a beta ratio and the two pressures you want. It
sizes the Venturi tube for you, builds the ISO 5167-4 geometry, meshes it,
runs an axisymmetric Navier-Stokes solve with turbulence, checks the answer
against the discharge coefficient in the standard, and writes out everything
from a DXF profile to a ParaView file.

No external CFD package. The solver is here, in about 900 lines of Python
with Numba-compiled kernels.

---

## Quick start

You need **Python 3.10 or newer**. Tested on 3.12.3, Linux.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python main.py                     # interactive wizard
python main.py --D 0.1 --beta 0.5 --p-inlet 101325 --p-throat 95000
python main.py --list-fluids
pytest tests/                      # 20 checks, a few seconds
```

The first run is slower than the rest: Numba compiles the kernels once and
caches them. Budget an extra half-minute the first time.

`cadquery` is deliberately not in `requirements.txt`. Without it you lose the
STEP export and nothing else - you still get the other nine files, and the
program tells you it skipped it.

### What comes out

`.vts` and `.vtp` for ParaView, `.dxf` of the 2D profile, `.stl` of the
revolved surface, two `.svg` figures (geometry and results), a `.ply` and
`.csv` point cloud, and a Markdown validation report. Optionally `.step` if
you install cadquery.

### How long it takes

Measured on a single CPU core:

| Grid | Iterations | Wall time |
|---|---|---|
| 90 x 36 (default) | 110,000 | ~1 min |
| 128 x 51 | 272,000 | ~4.5 min |

Yes, that is a lot of iterations. There is a reason, and it is honest - see
[Known limitation](#known-limitation) below.

---

## How it works

1. **Inverse sizing.** You impose the pressures; the code works backwards
   through Bernoulli to get throat velocity, flow rate and Reynolds number.
2. **ISO 5167-4 geometry.** 21 degrees total convergent angle over a length
   of 2.7*(D-d), a throat as long as its own diameter, a divergent between 7
   and 15 degrees, with the blend arcs the standard asks for.
3. **Finite-volume grid**, body-fitted, clustered towards the wall.
4. **Steady Navier-Stokes**, axisymmetric, with an algebraic turbulence
   model.
5. **Validation** against the standard's discharge coefficient.
6. **Export.**

---

## The design decisions worth explaining

These are the choices that make the numbers trustworthy. Each one exists
because the obvious alternative was tried and produced wrong answers.

### Finite volumes, not finite differences

The flux through each face is a single number shared by the two cells on
either side. When you add up the balance of every cell, the interior terms
cancel in pairs and all that survives is `Q_in - Q_out`.

That is the whole trick. Mass conservation is not something the solver tries
to achieve - it is an algebraic identity. Measured error between inlet and
outlet flow rate: around **1e-12 %**, which is round-off in the linear solver
and nothing else.

### One discrete operator, not two

The pressure Poisson matrix is assembled from *the same face coefficients*
that later correct the fluxes. There is no second discretisation to keep in
sync with the first.

This matters more than it sounds. A projection method is exact only when the
pressure matrix equals `D @ G` - the product of the discrete divergence and
gradient. If you build the Laplacian separately, using a slightly different
stencil, the projection stops removing the divergence and the solver quietly
converges to something that is not a solution.

### Geometric conservation law

Flux through the conical faces is not computed as `area * (u.n)` at the cell
centre. It is split into two exact geometric coefficients,
`Cr = 2*pi*eta*integral(R dz)` and `Cz = 2*pi*eta^2*integral(R*R' dz)`, the
second of which has the closed form `0.5*(R_right^2 - R_left^2)`.

Why bother: with the naive form, a perfectly **uniform** flow produces a
spurious divergence of order h^2 - and it does so precisely in the converging
and diverging cones, where the section changes. The projection has no way to
know that divergence is fake, so it invents a pressure field to cancel it.
With the split form, a uniform flow gives exactly zero divergence as an
algebraic identity. Measured: **2.8e-16**.

### Fluxes are the primary variable

Cell velocities are reconstructed from the fluxes, not the other way round.
This is the MAC arrangement: pressure at the centre, flux on the face, so the
pressure gradient correcting a flux only ever involves the two cells that
share it. Compact stencil, no odd-even checkerboard, no Rhie-Chow
stabilisation needed.

### Validate against C_d, not against ideal Bernoulli

Ideal Bernoulli ignores friction. A *correct* CFD result cannot match it -
the real pressure drop is always a few percent higher. So "agrees with
Bernoulli within 5%" is a broken acceptance test: it fails good answers and
passes answers that are wrong in the convenient direction.

The discharge coefficient is measured experimentally and tabulated by
ISO 5167-4, so it is an independent reference rather than a restatement of
the theory being checked. That is what the code compares against, at the tap
locations the standard specifies (0.5 D upstream of the convergent, and the
throat midpoint).

---

## What has actually been verified

| Property | Measured |
|---|---|
| Inlet/outlet mass balance | ~1e-12 % |
| Divergence after projection | machine precision |
| Geometric conservation law | ~1e-16 |
| Order of convergence (Hagen-Poiseuille) | **2.00** |
| dp/dz error at 80x40 (Poiseuille) | 0.031 % |
| Independence from the time step | identical between CFL 0.8 and 0.1 |
| Test suite | 20 / 20 |

The order of convergence and the pressure-gradient error come from a laminar
straight-pipe case, where the exact analytical solution is known. That is the
strongest check available: it exercises the viscous discretisation, the wall
gradient and the axisymmetric source term all at once, against an answer that
cannot be argued with.

The time-step independence check deserves a note, because it caught a real
bug. An earlier version accelerated the solve by treating radial diffusion
implicitly. It was about 7x faster and looked fine. It was not: halving the
CFL halved the error, four times running. An error that scales with the time
step is not a discretisation error - it means the *steady state itself*
depends on the time step, which means the converged answer is an artefact of
the scheme. Cause: the implicit preconditioner acted on the acceleration but
not on the pressure gradient, which arrives afterwards from the projection.
The two do not commute. Removing the accelerator took the error from 1.51%
to 0.031% and the order from 1.00 to 2.00.

### What has *not* been verified

- **Grid independence on the Venturi itself.** Two grid levels, not three.
  C_d moves from 0.9744 (90x36) to 0.9792 (128x51), which is the right
  direction, but two points do not demonstrate convergence.
- **A clean install from `requirements.txt` in a fresh environment.** Every
  listed package is confirmed to be imported by the code and nothing unlisted
  is required, but that is a static check, not an install test.

---

## Known limitation

The time step is capped by radial diffusion in the thin wall cells. Those
cells are thin on purpose (you want y+ around 1 to resolve the boundary
layer) and the eddy viscosity peaks there at ~200x the molecular value, so
the diffusive stability limit ends up roughly 30x tighter than the convective
one. Hence the six-figure iteration counts.

The implicit accelerator is still in the code:

```python
solve_fv(mesh, config, radial_implicit=True)   # ~7x fewer iterations
```

It is **off by default** for the reason described above: it makes the steady
state depend on the time step. Don't use it for anything you care about.

The proper fix is known and not exotic - a SIMPLE-style formulation with the
Poisson matrix weighted by 1/a_P, so that pressure and momentum pass through
the same operator. An incremental variant was attempted and turned out to be
unstable (the residual fell to 4.6e-4, then climbed back); under-relaxation
at 0.3 did not rescue it. That is the next piece of work, not a mystery.

---

## Approximations, stated plainly

- **Viscous term.** The code solves `div(nu_eff * grad(u))`. The transpose
  term `div(nu_eff * grad(u)^T)`, non-zero wherever the eddy viscosity
  varies, is omitted. Standard practice for algebraic-viscosity solvers; the
  contribution scales as `dnu_t/dr * dur/dz`, small both in the core and at
  the wall.
- **Turbulence.** Algebraic models only: Baldwin-Lomax (two-layer) and
  Prandtl mixing length with van Driest damping. No transport-equation model.
  Asking for one raises `NotImplementedError` rather than silently falling
  back to something else.
- **ISO validity range.** The standard defines C_d for
  2e5 <= Re_D <= 2e6. The default case runs at Re_D ~ 9.2e4, below that
  floor. The comparison still runs but is flagged as indicative rather than
  certifying. If you want a certifying comparison, raise the flow rate.

---

## Layout

```
main.py                     CLI and interactive wizard
venturi/
  config.py                 inverse sizing, fluid presets
  geometry.py               ISO 5167-4 wall profile
  fvmesh.py                 finite-volume grid, face areas, GCL coefficients
  projection.py             divergence-free projection, Poisson operator
  fvsolver.py               Navier-Stokes solver, Numba kernels
  turbulence.py             Baldwin-Lomax, mixing length
  validation_fv.py          C_d validation against ISO 5167-4
  optimizer.py              shape optimisation
  export.py                 output writers
  compat.py                 adapter for the export module
tests/                      conservation, physics, turbulence
```

---

## AI assistance disclosure

This project was developed with **Claude (Anthropic) used as a coding
assistant**. The AI contributed to the numerical scheme design, the
implementation, the diagnostic experiments that isolated the bugs described
above, and this documentation.

Everything in the "verified" table was measured by running the code, not
asserted. The reproduction paths are in `tests/`, and the ISO 5167-4
reference values were checked against sources rather than recalled from
memory. Known gaps are listed under "What has *not* been verified" and
"Known limitation" rather than glossed over.

Treat the results the way you would treat any in-house solver: the
conservation properties are provable and were measured, the Poiseuille
validation is against an exact solution, and the Venturi C_d comparison sits
outside the standard's validity range. Independent review is welcome.

---

## Licence

See `LICENSE`.
