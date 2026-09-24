# Venturi Design & CFD

[![License: MIT](https://img.shields.io/badge/License-MIT-blue)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)

A Venturi tube designer with its own flow solver attached.

A Venturi tube is a pipe that narrows and widens again. In the narrow part
(the throat) the fluid speeds up and its pressure drops, and that pressure
drop tells you the flow rate.

You give the program the pipe diameter, how much narrower the throat is, and
the two pressures you want. It works out the flow rate, draws the tube, cuts
the inside into small cells, solves the flow in it, checks whether the
result can be trusted, and writes drawings and data files into `results/`.

I wrote it because I wanted to understand projection methods by building
one, rather than by reading about one. There is no OpenFOAM or Fluent
underneath: the solver is plain Python with Numba for the heavy loops.

## Running it

Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt      # add requirements-dev.txt to run the tests

python main.py                       # interactive, asks you what you want
python main.py --D 0.1 --beta 0.5 --p-inlet 101325 --p-throat 95000
python main.py --list-fluids         # the fluid table
python main.py --help                # every option
python -m venturi.grid_study         # same case on 3 grids, error estimate
pytest tests/
```

The second line is the default case: water at 20 C, 100 mm pipe, 50 mm
throat, 6325 Pa pressure drop. It takes about 10 seconds. The very first run
takes about 30 seconds more while Numba compiles; after that it's cached.

The run ends with `RESULT: ALL CHECKS PASSED` (exit code 0) or
`RESULT: SOME CHECKS FAILED` (exit code 2). A wrong input stops it straight
away with `Error: ...` (exit code 1).

In interactive mode you either give D and beta yourself, or you give a flow
rate and a goal (smallest pressure loss, shortest tube, or a balance) and it
picks D, beta and the widening angle from quick hand formulas
(`venturi/optimizer.py`). The flow solver then checks the result. Press
Enter to accept the value in brackets.

## What comes out

One folder per case under `results/`, named after the inputs, so different
cases don't overwrite each other.

| file | what's in it | open with |
|---|---|---|
| `venturi_cfd_2d.vts` | speed and pressure on a flat slice through the axis | ParaView |
| `venturi_cfd_3d.vts` | the same slice spun around the axis into a 3D volume | ParaView |
| `venturi_profile.dxf` | the wall line, top and bottom, and the axis | any CAD program |
| `venturi_3d.stl` | the inside wall as a triangle surface | CAD, 3D viewers |
| `venturi_3d.step` | the fluid volume as a solid, only if `cadquery` is installed | CAD |
| `venturi_geometry.svg` | drawing of the profile | browser |
| `venturi_results.svg` | speed map, pressure map, pressure along the axis | browser |
| `venturi_pointcloud.csv` | `x, y, z, pressure, velocity_magnitude`, one point per row | spreadsheet, Python |
| `venturi_pointcloud.ply` | the same points for 3D viewers | ParaView, MeshLab |
| `venturi_report.md` | inputs, geometry, every check | text editor |

The tube axis is `x` and the flow goes towards `+x`. The field files hold
cell-centre values, so they stop half a cell short of the wall and the axis.

`cadquery` isn't in requirements.txt on purpose. Install it if you want the
STEP file; without it you get the other nine files and a warning. The same
goes for `pyvista` and `ezdxf`: if one is missing, the files that need it
are skipped with a warning and are not listed as written.

The grid study writes `grid_study.md` and `grid_study.csv` to
`results/grid_study_<case>/`.

## Options

Lengths in metres, pressures in pascal (101325 Pa is normal air pressure at
sea level), angles in degrees.

**Fluid**

| option | meaning | default |
|---|---|---|
| `--fluid NAME` | a fluid from `--list-fluids`, or `custom` | `water_20C` |
| `--rho`, `--mu` | density [kg/m3] and viscosity [Pa s], needed with `custom` | - |
| `--p-vap` | vapour pressure [Pa] for `custom`; 0 switches the cavitation check off | 0 |

**Tube**

| option | meaning | default |
|---|---|---|
| `--D` | inside diameter of the pipe | 0.1 |
| `--beta` | throat diameter / pipe diameter, between 0 and 1 | 0.5 |
| `--alpha-conv` | full angle of the narrowing cone | 21 |
| `--alpha-div` | full angle of the widening cone | 8 |
| `--no-blend` | sharp corners instead of rounded ones | rounded |

**Operating point**

| option | meaning | default |
|---|---|---|
| `--p-inlet` | pressure before the tube | 101325 |
| `--p-throat` | pressure you want in the throat, lower than `--p-inlet` | 95000 |
| `--cd-design` | discharge coefficient assumed when sizing (1 = no friction) | 1.0 |

**Simulation**

| option | meaning | default |
|---|---|---|
| `--Nz`, `--Nr` | cells along the tube, and from the axis to the wall | 90, 36 |
| `--clustering` | how much thinner the cells get at the wall (0 = all equal) | 2.7 |
| `--throat-refine` | how many times shorter the throat cells are: `auto` = 1/beta^2 (at most 6), or a number | `auto` |
| `--turbulence` | `laminar`, `mixing_length` or `baldwin_lomax` | `baldwin_lomax` |
| `--time-step` | `local`: each cell takes its own safe step; `global`: all take the smallest | `local` |
| `--cfl` | time step safety factor, below 1; lower is slower but safer | 0.6 |
| `--tol` | stop when the residual is below this | 1e-3 |
| `--max-iter` | give up after this many steps | 400000 |

**Checks and output**

| option | meaning | default |
|---|---|---|
| `--cd-ref`, `--cd-tol` | compare C_d with a value of yours (from a datasheet or a test), within this % | none, 3 |
| `--output-dir` | where the files go | `results/<name from the inputs>` |
| `--no-export` | compute only, write nothing | off |

The grid study takes all of these plus `--levels` (grids as `NzxNr`, comma
separated, default `64x26,90x36,128x51`) and `--jobs` (grids solved at the
same time). Its residual target is 1e-4.

## Reading a run

The screen shows six steps.

**1. Sizing.** From the two pressures, Bernoulli plus conservation of mass
give the speeds and the flow rate:

    v_throat = C_design * sqrt( 2 * (p_inlet - p_throat) / (rho * (1 - beta^4)) )
    v_inlet  = v_throat * beta^2

With `C_design = 1` that's frictionless flow, so the simulated pressure drop
comes out a few percent larger than the one you asked for. If you already
know your tube's C_d, pass it with `--cd-design`. For a gas the density is
taken at `--p-inlet`, since a gas at 2 bar is twice as dense as at 1 bar.
Warnings follow the summary: laminar/turbulent mismatch, throat pressure
below the vapour pressure, gas Mach number above 0.3.

**2. Wall profile.** Straight pipe (1.5 D), narrowing cone, throat (length
d), widening cone, straight pipe (3 D). The four corners are rounded with
arcs that meet the straight pieces without a kink.

**3. Grid.** Ring-shaped cells, thinner towards the wall because that's
where the speed changes fastest, and shorter in the throat (see below).

**4. Flow solution.** A line every 5000 steps:

    iter  5000 | residual 5.5e-02 | dt 6.68e-05..3.77e-03 s | max nu_t/nu 594.7 | matrix refactored 298x

`residual` is how much the flow still changes; the run stops below `--tol`.
`dt` is the smallest and largest time step among the cells. `nu_t/nu` is how
much stronger turbulent mixing is than plain viscosity, at its peak.
`matrix refactored` counts how often the pressure equations were rebuilt,
which happens when the time steps are updated.

**5. Checks.**

| check | what it means | passes when | if it fails |
|---|---|---|---|
| Inlet/outlet mass balance | what goes out equals what comes in | mismatch < 0.1 % | a bug, tell me |
| Leftover divergence | no cell makes or loses fluid | < 1e-9 of the flow through the cell | a bug, tell me |
| Steady state reached | the flow stopped changing | residual < `--tol` | see the last section |
| First cell y+ | wall cells thin enough for the turbulence model | y+ < 5 | raise `--clustering` or `--Nr` |
| No cavitation | lowest pressure stays above the vapour pressure, so the liquid doesn't boil | p_min > p_vap | raise `--p-throat` or `beta` |
| C_d vs reference | only with `--cd-ref` | within `--cd-tol` | - |

Then the numbers: C_d, the pressure drop between the wall taps, the same
drop averaged over the cross-section (for comparison), the frictionless
drop, and how much of the drop is recovered in the widening cone.

**6. Files.** See [What comes out](#what-comes-out).

## Why the solver is built this way

Most of these choices only make sense once you've seen the alternative fail,
so I've written down what went wrong rather than just what the code does.

**Finite volumes rather than finite differences.** The flux through a face is
one number, shared by the two cells on either side. Add up every cell's
balance and the interior terms cancel in pairs, leaving only `Q_in - Q_out`.
So mass conservation isn't something the solver works towards, it's just
algebra.

**The pressure matrix is assembled from the same face coefficients that
correct the fluxes.** A projection (the step that makes every cell's inflow
equal its outflow, and gives the pressure as a by-product) is exact only if
the pressure matrix is literally `D @ G`, divergence times gradient. Build
the Laplacian separately with a slightly different stencil and the
projection stops removing divergence, but everything still runs and
converges to something plausible-looking. This is the failure mode I started
from, and it's nasty precisely because nothing crashes.

**Cone faces use exact geometric coefficients** (`Cr`, `Cz`) instead of
`area * (u.n)` at the cell centre. With the naive version a perfectly
uniform flow generates a fake divergence of order h^2, right in the cones,
exactly where you care. The projection can't tell it's fake and invents
pressure to cancel it. With the split form, uniform flow gives zero
divergence identically.

**Fluxes are the primary variable**, cell velocities are rebuilt from them.
That's the MAC layout, so the pressure gradient correcting a flux only
touches the two cells next to it. No checkerboard, no Rhie-Chow.

**C_d, not Bernoulli.** Bernoulli ignores friction, so a correct result
can't match it: the real pressure drop is always a few percent higher.
"Within 5 % of Bernoulli" fails good answers and passes answers that are
wrong in the convenient direction. C_d is the real flow rate divided by the
frictionless one for the same measured drop, so it says how far from ideal
the tube is. The program always computes it; `--cd-ref` compares it with a
value of yours.

**Taps read the wall.** A real tube measures pressure through small holes
in the wall, 0.5 D before the narrowing cone and in the middle of the
throat, so the program reads the wall cells there, interpolated to the exact
position. I first took the section average at the nearest cell. In the
throat that's wrong twice: the pressure isn't uniform across the section
(the flow is still curving after the corner), and it changes fast along the
axis, so the nearest cell could be half a cell off. With 4 cells in the
throat, C_d moved by 0.7 % depending on where the tap landed.

**Shorter cells in the throat.** That's also why the throat now gets
shorter cells: 1/beta^2 times shorter by default. The fluid is 1/beta^2
times faster there, so this keeps the time it takes to cross one cell the
same in the throat and in the pipe, which means the throat never forces a
smaller time step (same CFL). The cell length changes by at most 20 % from
one cell to the next.

**Each cell takes its own time step.** The thin wall cells, where eddy
viscosity reaches about 600 times the plain one, limit the step to about
7e-5 s. The cells in the middle could take 3e-3 s. With one step for all,
the default case needed about 110,000 iterations and the 128 x 51 grid
didn't settle at all. Only the final settled flow matters, so each cell may
move with its own largest safe step. The catch is that the answer mustn't
depend on the steps, and an earlier shortcut of mine did exactly that: it
treated radial diffusion implicitly, ran 7 times faster, looked fine, and
then I halved the CFL and the error halved too, four times in a row. Error
that scales with dt means the converged solution itself depends on the time
step. That code is gone.

The local step avoids it by putting the step into both halves of the
update. Each face uses the smaller step of its two cells, in the prediction
and in the pressure correction, and the pressure matrix is weighted by the
same steps. When nothing changes any more, the step multiplies zero and
drops out. The tests check it: same answer at CFL 0.6 and 0.15, and same
answer with local and global steps. The global mode is still there
(`--time-step global`) and gives the old solver's result bit for bit.

## What I checked

| | |
|---|---|
| Mass balance in/out, default case | ~3e-13 % |
| Divergence after projection | ~2.5e-13 (round-off) |
| Uniform flow through the cones, fake divergence | < 1e-13 |
| dp/dz error, 80 x 40 Poiseuille | 0.031 % |
| Order of convergence, Poiseuille 20x10 / 40x20 / 80x40 | 2.0 |
| Steady state vs time step, CFL 0.6 vs 0.15 | same to 1e-6 |
| Local vs global time step | same to 1e-6 |
| Tests | 72 |

The Poiseuille numbers come from laminar flow in a straight pipe, where
Hagen-Poiseuille gives the exact answer. It's the best check I have, because
it hits the viscous terms, the wall gradient and the axisymmetric source all
at once against something that can't be argued with.

**Grid study on the Venturi.** Default case, four grids, each with about
sqrt(2) times more cells per direction, same spacing shape, residual target
1e-4:

| grid | settled | y+ | C_d |
|---|---|---|---|
| 64 x 26 | yes | 6.0 | 0.97003 |
| 90 x 36 (default) | yes | 4.2 | 0.97491 |
| 128 x 51 | yes | 3.1 | 0.97737 |
| 180 x 72 | no | 2.1 | 0.97839 |

Each refinement adds about half of what the previous one did (+0.0049,
+0.0025, +0.0010), which is what a second-order scheme should do. From the
three settled grids: observed order 2.16, extrapolated C_d 0.9795 for an
infinitely fine grid, and a GCI of 0.28 % on the 128 x 51 value, i.e. the
grid-independent C_d should lie within 0.28 % of 0.9774. The default grid
reads about 0.47 % low; use 128 x 51 (about 35 s) when that matters.
(Extrapolation and GCI follow Celik et al., J. Fluids Eng. 130, 2008.)

The 180 x 72 grid never settles, its residual hovers near 1e-2, but its C_d
came out 0.97839 to five digits in two runs stopped at different moments,
and it's where the trend predicts (0.97854). So the hovering doesn't move
C_d.

## What I have not shown, and the approximations

- The finest grids aren't fully settled, so the error estimate rests on
  64 x 26, 90 x 36 and 128 x 51, and 64 x 26 has y+ = 6, a bit above what
  the wall models need.
- Density is constant. For gases that only holds at low Mach number, so the
  program warns above Ma = 0.3.
- Turbulence is algebraic only: Baldwin-Lomax and mixing length with van
  Driest damping. No transport-equation model. Asking for one stops the
  program instead of quietly using something else.
- The viscous term is `div(nu_eff * grad u)`. The transpose term is
  dropped, which is the usual thing to do with an algebraic eddy viscosity.
- On the cone faces the line between two cell centres isn't quite normal to
  the face (up to about 10 degrees). The pressure correction uses only the
  aligned part. Mass is still exact, but the pressure picks up a small error
  there that shrinks with the grid. It's probably behind the small wobble
  (about 40 Pa) of the wall pressure just after the throat corner.
- No cavitation model, no swirl, no heat.
- Not covered by the tests: the STEP export (it needs `cadquery`) and the
  interactive mode.

## Speed

One CPU core, `--tol 1e-3`:

| case | global step | local step | C_d global / local |
|---|---|---|---|
| default, 90 x 36 | 116,891 steps, 44 s | 18,556 steps, 11 s | 0.97503 / 0.97510 |
| 128 x 51 | doesn't settle in 400,000 | 34,061 steps, 34 s | - / 0.97728 |
| beta = 0.4 | 216,695 steps, 81 s | 19,386 steps, 10 s | 0.97146 / 0.97145 |
| beta = 0.7 | 63,686 steps, 24 s | 33,011 steps, 14 s | 0.97909 / 0.97932 |
| air, 1325 Pa drop | doesn't settle in 400,000 | 16,115 steps, 9 s | - / 0.97429 |
| mixing length model | 125,136 steps, 45 s | 19,931 steps, 9 s | 0.97353 / 0.97351 |

Where both settle, they agree on C_d within 0.025 %.

The local step doesn't always win. In a straight pipe with equal cells every
cell has nearly the same limit, so there's nothing to gain and it takes
about 1.5 times more steps (it keeps a 20 % safety margin). On very coarse
grids (40 x 16, y+ = 8.7) and on 180 x 72 it keeps hovering instead of
settling: Baldwin-Lomax jumps between neighbouring cells and the bigger
steps keep that going. Freezing the eddy viscosity makes it settle, which is
how I know that's the cause. Use `--time-step global` or a different grid
there.

## Symbols

| symbol | meaning | unit |
|---|---|---|
| D, d | pipe and throat diameter, d = beta * D | m |
| beta | d / D | - |
| alpha_conv, alpha_div | full angle of the narrowing and widening cone | deg |
| z, r | distance along the axis, distance from the axis | m |
| R(z) | wall radius at z | m |
| eta | r / R(z): 0 on the axis, 1 at the wall | - |
| u_z, u_r | speed along the axis and towards the wall | m/s |
| p, p_vap | pressure, vapour pressure (below it a liquid boils) | Pa |
| rho, mu, nu | density, viscosity, nu = mu / rho | kg/m3, Pa s, m2/s |
| nu_t | eddy viscosity: extra mixing from turbulence, treated as extra viscosity | m2/s |
| Q | flow rate | m3/s |
| Re_D | Reynolds number rho v D / mu; below ~2300 smooth (laminar), above ~4000 turbulent | - |
| C_d | discharge coefficient: Q / (A_throat sqrt(2 dp / (rho (1 - beta^4)))) | - |
| sigma | cavitation number (p_throat - p_vap) / (rho v_throat^2 / 2); small means close to boiling | - |
| Ma | Mach number, speed / speed of sound | - |
| y+ | distance of the first cell centre from the wall, scaled by the wall friction | - |
| CFL | time step x speed / cell size, must stay below 1 | - |
| dt | time step | s |
| residual | largest change of speed per unit time, scaled by tube length / throat speed^2 | - |
| h, p, GCI | grid study: cell size (1/sqrt(cells)), observed order (error goes like h^p), error band on the finest result | -, -, % |

## Layout

```
main.py                    starts the program
venturi/
  cli.py                   options, interactive mode, the six steps
  fluids.py                fluid table, gas density vs pressure
  config.py                inputs, sizing, warnings
  geometry.py              wall profile with rounded corners
  fvmesh.py                cells, face areas, volumes, throat refinement
  projection.py            pressure correction that keeps mass exact
  solver/
    solver.py              the time loop, local or global step
    kernels.py             the heavy loops, compiled with Numba
  turbulence.py            Baldwin-Lomax, mixing length
  validation.py            checks, wall taps, C_d
  grid_study.py            same case on several grids, error estimate
  optimizer.py             quick choice of D, beta, angle (interactive mode 2)
  export/                  paraview.py, cad.py, plots.py, tables.py
tests/
results/                   made by the runs, not in git
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest tests/          # 72 tests, about 10 seconds
```

| file | what it checks |
|---|---|
| `test_conservation.py` | volumes add up; no fake mass in uniform flow; the projection removes all imbalance, leaves inlet and wall alone, and its matrix is symmetric, also with a different time step on every face and on the refined throat grid |
| `test_physics.py` | Poiseuille pressure gradient and profile; same answer at different CFL and with local vs global step; inlet profile |
| `test_turbulence_models.py` | eddy viscosity never negative, zero for laminar, large at high Re; unknown models refused |
| `test_geometry.py` | radius and slope continuous at the rounded corners |
| `test_inputs.py` | bad inputs refused; sizing; gas density; Mach and cavitation warnings; options reach the program; throat refinement rule; optimiser; wall taps |
| `test_grid_study.py` | extrapolation and GCI recover a known power law exactly; up-and-down results flagged |
| `test_export.py` | every file puts the right value on the right point; missing libraries skip cleanly |

## When something fails

If "Steady state reached" fails, give it more steps with `--max-iter`. If the
residual stops going down and just hovers, `--time-step global` or a lower
`--cfl` (0.4) usually gets it there; see [Speed](#speed).

If "First cell y+" fails, the wall cells are too thick for the turbulence
model. Raise `--clustering` (3.0) or `--Nr`.

A Mach warning with a gas means the pressure drop is too big for the inlet
pressure. The gas density then changes along the tube while the program
keeps it fixed, so treat the result as indicative.

For a fluid that isn't in the list:
`--fluid custom --rho 850 --mu 0.003 --p-vap 500`

If you spot something wrong, tell me.

## Licence

MIT, see LICENSE.
