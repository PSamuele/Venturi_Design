# Venturi Design & CFD

[![License: MIT](https://img.shields.io/badge/License-MIT-blue)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)

A small program that **designs a Venturi tube** and then **checks the design
with its own flow simulation**.

A Venturi tube is a pipe that narrows and then widens again. Where it is
narrow (the *throat*) the fluid goes faster and its pressure drops. You
measure that pressure drop and it tells you how much fluid is flowing.

You give the program the pipe size, how much narrower the throat should be,
and the two pressures you want. It works out the flow rate, draws the tube,
simulates the flow inside it, tells you whether the result can be trusted,
and saves drawings and data files into the `results/` folder.

---

## Contents

1. [Install](#1-install)
2. [Quick start](#2-quick-start)
3. [All command-line options](#3-all-command-line-options)
4. [Interactive mode](#4-interactive-mode)
5. [What happens during a run](#5-what-happens-during-a-run)
6. [Output files](#6-output-files)
7. [Reading the checks](#7-reading-the-checks)
8. [Symbols](#8-symbols)
9. [Words used in this README](#9-words-used-in-this-readme)
10. [How accurate it is, and its limits](#10-how-accurate-it-is-and-its-limits)
11. [Speed](#11-speed)
12. [Project layout](#12-project-layout)
13. [Tests](#13-tests)
14. [Questions and problems](#14-questions-and-problems)

---

## 1. Install

You need Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate            # on Windows: .venv\Scripts\activate
pip install -r requirements.txt      # to run the program
pip install -r requirements-dev.txt  # only if you also want to run the tests
```

`cadquery` is optional and not in the list. Install it only if you want the
3D solid as a `.step` file. Without it you get every other file and one
warning line.

The first run takes about 30 extra seconds, because Numba translates the
heavy loops into machine code. That code is saved, so later runs start at
once.

## 2. Quick start

```bash
python main.py                                        # asks you questions
python main.py --D 0.1 --beta 0.5 --p-inlet 101325 --p-throat 95000
python main.py --list-fluids                          # show the fluid table
python main.py --help                                 # show every option
```

The second line runs the default case: water at 20 C, a 100 mm pipe, a
50 mm throat, and a pressure drop of 6325 Pa. It takes about 40 seconds and
writes its files to
`results/water_20C_D100_beta0.50_Pin101325_Pth95000/`.

The program ends with one of these lines:

- `RESULT: ALL CHECKS PASSED` (exit code 0)
- `RESULT: SOME CHECKS FAILED` (exit code 2; see [section 7](#7-reading-the-checks))

A wrong input stops it straight away with `Error: ...` (exit code 1).

## 3. All command-line options

Units: lengths in metres, pressures in pascal (Pa), angles in degrees.
101325 Pa is normal air pressure at sea level.

**Fluid**

| option | meaning | default |
|---|---|---|
| `--fluid NAME` | a fluid from the table (`--list-fluids`), or `custom` | `water_20C` |
| `--rho VALUE` | density in kg/m3. Needed with `custom` | - |
| `--mu VALUE` | dynamic viscosity in Pa s. Needed with `custom` | - |
| `--p-vap VALUE` | vapour pressure in Pa, for `custom` (0 = no cavitation check) | 0 |
| `--list-fluids` | print the fluid table and stop | - |

**Tube shape**

| option | meaning | default |
|---|---|---|
| `--D VALUE` | inside diameter of the pipe, before and after the tube | 0.1 |
| `--beta VALUE` | throat diameter divided by pipe diameter, between 0 and 1 | 0.5 |
| `--alpha-conv VALUE` | full opening angle of the narrowing cone | 21 |
| `--alpha-div VALUE` | full opening angle of the widening cone | 8 |
| `--no-blend` | sharp corners instead of rounded ones | rounded |

**Operating point**

| option | meaning | default |
|---|---|---|
| `--p-inlet VALUE` | pressure before the tube | 101325 |
| `--p-throat VALUE` | pressure you want in the throat. Must be lower than `--p-inlet` | 95000 |
| `--cd-design VALUE` | discharge coefficient assumed when sizing (see [section 5](#5-what-happens-during-a-run)) | 1.0 |

**Simulation**

| option | meaning | default |
|---|---|---|
| `--Nz N` | number of cells along the tube | 90 |
| `--Nr N` | number of cells from the axis to the wall | 36 |
| `--clustering VALUE` | how much thinner the cells get near the wall (0 = all the same) | 2.7 |
| `--throat-refine VALUE` | how many times shorter the cells are in the throat than far from it (1 = all the same length) | 4 |
| `--turbulence MODEL` | `laminar`, `mixing_length` or `baldwin_lomax` | `baldwin_lomax` |
| `--max-iter N` | give up after this many steps | 400000 |
| `--tol VALUE` | the solution counts as settled when the residual is below this | 1e-3 |
| `--cfl VALUE` | safety factor for the time step, between 0 and 1. Lower is slower but safer | 0.6 |

**Checks and output**

| option | meaning | default |
|---|---|---|
| `--cd-ref VALUE` | a C_d to compare with, for example from a datasheet or a test | none |
| `--cd-tol VALUE` | allowed difference from `--cd-ref`, in % | 3 |
| `--output-dir PATH` | where to write the files | `results/<name from the inputs>` |
| `--no-export` | compute only, write no files | off |

## 4. Interactive mode

`python main.py` with no options asks a few questions. Press Enter to accept
the value in brackets. A value that is not valid is asked again.

- **Mode 1**: you choose `D` and `beta` yourself.
- **Mode 2**: you give a flow rate and a goal, and the program picks `D`,
  `beta` and the widening angle. Goal 1 is the smallest pressure loss, goal
  2 the shortest tube, goal 3 a balance of the two. This choice uses quick
  hand formulas, not the simulation (see `venturi/optimizer.py`). The
  simulation then checks the result.

## 5. What happens during a run

The screen shows six numbered steps.

**[1/6] Sizing.** From the two pressures the program works out the speeds
and the flow rate with Bernoulli's equation plus conservation of mass:

    v_throat = C_design * sqrt( 2 * (p_inlet - p_throat) / (rho * (1 - beta^4)) )
    v_inlet  = v_throat * beta^2
    Q        = v_inlet * pi * D^2 / 4

With `C_design = 1` this is the flow with no friction at all. A real tube
loses a little energy, so the simulated pressure drop comes out a few
percent larger than the one you asked for. If you already know the C_d of
your tube, give it with `--cd-design` to compensate.

For a gas, the density is taken at `--p-inlet` (a gas at 2 bar is twice as
dense as at 1 bar). The summary also prints Re_D, the cavitation number and,
for gases, the Mach number (see [section 8](#8-symbols)). The warnings
below it explain any problem found.

**[2/6] Wall profile.** The tube is built from five pieces:

    straight pipe (1.5 D) | narrowing cone | throat (length d) | widening cone | straight pipe (3 D)

The lengths of the cones follow from the angles. The four corners are
rounded with circular arcs that meet the straight pieces smoothly.

**[3/6] Grid.** The inside of the tube is cut into ring-shaped cells, Nz
along the axis and Nr from the axis to the wall. The cells get thinner near
the wall, because that is where the speed changes fastest. They are also
shorter in the throat (4 times by default), because the pressure changes
quickly there: it dips just after the throat entrance, where the flow turns
round the corner, and the throat pressure tap sits right after that dip.
The cell length changes gradually, by at most about 20 % from one cell to
the next.

**[4/6] Flow solution.** The program solves the flow equations of a
liquid, or of a slow gas, in the tube (see *Navier-Stokes* in
[section 9](#9-words-used-in-this-readme)). It starts from a rough guess
and takes small time steps until nothing changes any more. A line is
printed every 5000 steps:

    iter  5000 | residual 6.7e-01 | dt 8.3e-05 s | max nu_t/nu 532.0

- `residual` measures how much the flow still changes. The run stops when
  it drops below `--tol`.
- `dt` is the time step.
- `max nu_t/nu` is how many times stronger turbulent mixing is than plain
  viscosity, at its peak.

**[5/6] Checks.** See [section 7](#7-reading-the-checks).

**[6/6] Files.** See [section 6](#6-output-files).

## 6. Output files

Everything goes into one folder under `results/`. The folder name is built
from the inputs, so different cases do not overwrite each other.

| file | what it holds | open with |
|---|---|---|
| `venturi_cfd_2d.vts` | speed and pressure on a flat slice through the axis | ParaView |
| `venturi_cfd_3d.vts` | the same slice spun around the axis into a full 3D volume | ParaView |
| `venturi_profile.dxf` | the wall line (top and bottom) and the axis | any CAD program |
| `venturi_3d.stl` | the inside wall as a triangle surface | CAD, 3D viewers, slicers |
| `venturi_3d.step` | the fluid volume as a solid. Only if `cadquery` is installed | CAD |
| `venturi_geometry.svg` | drawing of the profile | web browser |
| `venturi_results.svg` | speed map, pressure map, pressure along the axis | web browser |
| `venturi_pointcloud.csv` | one row per point: `x, y, z, pressure, velocity_magnitude` | spreadsheet, Python |
| `venturi_pointcloud.ply` | the same points for 3D viewers | ParaView, MeshLab |
| `venturi_report.md` | inputs, geometry and every check, as a table | any text editor |

In all 3D files the tube axis is `x`, and the flow goes towards `+x`. The
field files contain the values at the cell centres, so they stop half a
cell short of the wall and of the axis.

If `pyvista` or `ezdxf` is missing, the files that need it are skipped with
a warning and are not listed as written.

## 7. Reading the checks

Every check prints `PASS` or `FAIL`, its value and its target.

| check | what it means | passes when | if it fails |
|---|---|---|---|
| Inlet/outlet mass balance | the flow rate going out equals the flow rate coming in | mismatch < 0.1 % | a bug; please report it |
| Leftover divergence | no cell creates or loses fluid | < 1e-9 of the flow through the cell | a bug; please report it |
| Steady state reached | the flow stopped changing before `--max-iter` | residual < `--tol` | raise `--max-iter`, or read [section 11](#11-speed) |
| First cell y+ | the cells touching the wall are thin enough for the turbulence model | y+ < 5 | raise `--clustering` or `--Nr` |
| No cavitation | the lowest pressure anywhere stays above the vapour pressure, so the liquid does not boil | p_min > p_vap | raise `--p-throat`, or use a larger `beta` |
| C_d vs reference | only with `--cd-ref` | within `--cd-tol` % | see below |

Below the checks you get these numbers:

- **C_d (computed)**: the discharge coefficient (see
  [section 8](#8-symbols)). Usually a little below 1.
- **dp between the wall taps**: the pressure drop used for C_d. It is read
  at the wall, 0.5 D before the narrowing cone and in the middle of the
  throat, the same places a real tube has its small pressure holes. The
  wall values of the two nearest cells are interpolated to those exact
  positions.
- **dp of section averages**: the same drop averaged over the whole cross
  section, only for comparison. It differs from the wall value where the
  flow is curved (see [section 10](#10-how-accurate-it-is-and-its-limits)).
- **dp without friction**: what Bernoulli predicts, only for comparison.
- **pressure recovered**: how much of the drop comes back in the widening
  cone. The rest is lost for good.

## 8. Symbols

| symbol | name | unit | meaning |
|---|---|---|---|
| D | pipe diameter | m | inside diameter before and after the tube |
| d | throat diameter | m | d = beta * D |
| beta | diameter ratio | - | d / D |
| alpha_conv | narrowing angle | deg | full opening angle of the narrowing cone |
| alpha_div | widening angle | deg | full opening angle of the widening cone |
| z | axial position | m | distance along the axis from the start of the tube |
| r | radius | m | distance from the axis |
| R(z) | wall radius | m | radius of the wall at position z |
| eta | relative radius | - | r / R(z): 0 on the axis, 1 at the wall |
| u_z, u_r | velocity parts | m/s | speed along the axis and towards the wall |
| p | pressure | Pa | absolute static pressure |
| p_vap | vapour pressure | Pa | below this a liquid starts to boil |
| rho | density | kg/m3 | mass per volume |
| mu | dynamic viscosity | Pa s | how "thick" the fluid is |
| nu | kinematic viscosity | m2/s | mu / rho |
| nu_t | eddy viscosity | m2/s | extra mixing caused by turbulence (see section 9) |
| Q | flow rate | m3/s | volume passing per second |
| Re_D | Reynolds number | - | rho * v_inlet * D / mu. Below about 2300 the flow is smooth (laminar), above about 4000 it is turbulent |
| C_d | discharge coefficient | - | real flow rate divided by the frictionless one for the same measured pressure drop: Q / (A_throat * sqrt(2 dp / (rho (1 - beta^4)))) |
| sigma | cavitation number | - | (p_throat - p_vap) / (rho v_throat^2 / 2). The smaller it is, the closer the liquid is to boiling |
| Ma | Mach number | - | speed / speed of sound. Above 0.3 a gas can no longer be treated as having constant density |
| y+ | wall distance in wall units | - | distance of the first cell centre from the wall, scaled by the friction at the wall |
| CFL | Courant number | - | time step times speed divided by cell size; must stay below 1 |
| dt | time step | s | how far the simulation moves in one step |
| residual | residual | - | largest change of speed per step, divided by dt and made dimensionless with the tube length and the throat speed |

## 9. Words used in this README

- **Laminar / turbulent**: in laminar flow the fluid moves in smooth
  layers. In turbulent flow it swirls in eddies of many sizes, and those
  eddies mix it far more strongly. Most pipes in practice run turbulent.
- **Navier-Stokes equations**: the basic equations of fluid motion (mass is
  conserved, and force equals mass times acceleration). This program solves
  them assuming the tube is round (*axisymmetric*: every slice through the
  axis looks the same) and the density is constant.
- **Finite volumes**: the tube is cut into small cells. For each cell the
  program keeps track of what flows in and out through its faces. A face is
  shared by two cells, so what leaves one enters the other exactly, and no
  fluid can be lost.
- **Projection**: after each step the program corrects the flow so that
  every cell has exactly as much coming in as going out. The pressure
  comes out of that correction.
- **Eddy viscosity (nu_t)**: a way to account for turbulence without
  simulating every eddy. The eddies' mixing is treated as extra viscosity.
  *Mixing length* and *Baldwin-Lomax* are two recipes for computing it
  from the local flow.
- **Boundary layer**: the thin layer next to the wall where the fluid slows
  from full speed down to zero. Most of the friction happens there.
- **Wall clustering**: making the cells thinner near the wall so the
  boundary layer is described well.
- **Steady state**: the flow no longer changes with time. The program
  steps forward in time only to reach it.
- **Cavitation**: when the pressure of a liquid drops below its vapour
  pressure, bubbles of vapour form and then collapse. This damages the
  tube, and the program cannot simulate it.

## 10. How accurate it is, and its limits

**What was checked, by running the code:**

| check | result |
|---|---|
| inlet/outlet mass balance, default case | 3e-13 % |
| leftover divergence after each step | 2.5e-13 (round-off) |
| uniform flow through the cones creates no fake mass | < 1e-13 |
| pressure gradient in a straight pipe vs the exact laminar result (Hagen-Poiseuille), 80 x 40 cells | 0.031 % error |
| how that error shrinks with the grid (20x10, 40x20, 80x40) | order 2.0: half the cell size, one quarter of the error |
| solution independent of the time step (CFL 0.6 vs 0.15) | same to 1e-6 |

The straight-pipe test is the strongest one. There the exact answer is
known, and the test involves the viscous terms, the wall treatment and the
round geometry all at once.

**Default Venturi case (90 x 36, throat refinement 4):** C_d = 0.9749. The
wall pressure drop is 6655 Pa, against 6325 Pa without friction.

**How C_d depends on the throat cells** (default case, taps interpolated to
their exact position):

| throat refinement | cells in the throat | C_d |
|---|---|---|
| 1 (even spacing) | 4 | 0.9707 |
| 2 | 7 | 0.9751 |
| 3 | 9 | 0.9764 |
| 4 (default) | 12 | 0.9751 |

With even spacing, the pressure dip at the throat entrance falls inside a
single cell, so the throat reading is off by about 0.4 %. From refinement 2
upwards C_d stays within about +/- 0.07 %. The spread that remains comes
from small ups and downs of the wall pressure (about +/- 40 Pa) just after
the sharp corner at the throat entrance. The four rounded corners are
genuinely small here: with radius 0.2 d and a 10.5 degree turn, the arc
into the throat is only 1.8 mm long.

Tightening `--tol` from 1e-3 to 1e-4 changes C_d by at most 0.035 % and
takes about three times longer.

**What has NOT been shown:**

- **That the Venturi result is grid independent.** On the finer 128 x 51
  grid the run does not settle within 400,000 steps (see
  [section 11](#11-speed)), so there is no converged second grid to compare
  with.
- **That C_d is settled to better than about 0.1 %.** See the table above:
  the wall pressure right after the throat corner still wobbles a little
  from cell to cell. The pressure also differs between axis and wall in the
  throat (about 70 Pa, 1 % of dp), because the flow is still curving after
  the corner. That is why the wall reading and the section average differ.

**Simplifications in the model:**

- Density is constant. For gases that holds only at low Mach number: the
  program warns above Ma = 0.3.
- Turbulence uses algebraic models only (mixing length, Baldwin-Lomax).
  There is no model with transport equations. Asking for another model
  stops the program with an error instead of quietly using something else.
- The viscous term is `div(nu_eff * grad u)`. The part `div(nu_eff * grad u^T)`
  is left out, as usual with algebraic eddy viscosity.
- On the cone faces the cell centres are not exactly in line with the face
  normal (up to about 10 degrees). The pressure correction uses only the
  aligned part. Mass is still conserved exactly, but the pressure picks up
  a small error there that shrinks as the grid is refined.
- No cavitation model, no swirl, no heat.

## 11. Speed

Measured on one CPU core of the machine used for development:

| grid | steps | time | settles? |
|---|---|---|---|
| 90 x 36 (default) | 108,411 | about 40 s | yes |
| 128 x 51 | 400,000 (limit) | about 5 min | no, residual stalls near 2e-2 |

The 128 x 51 case also stalls with the code as it was before this
version (checked: same residual, 2.1e-2, after 400,000 steps). An earlier
README said it settles in about 272,000 steps, but the code in the
repository does not reproduce that.

Why so many steps: the cells at the wall are very thin, and the eddy
viscosity there reaches about 600 times the plain viscosity. Together these
force a very small time step for the diffusion near the wall. That limit is
tighter than the one set by the flow speed.

There is a faster option in the code,
`solve_fv(mesh, config, radial_implicit=True)`, which needs about 7 times
fewer steps. It is off because with it the final answer depends on the time
step, which is wrong. Do not use it for results you care about.

The real fix is a different time-marching method (SIMPLE-type, or a time
step chosen cell by cell). That is future work.

## 12. Project layout

```
main.py                    starts the program (calls venturi/cli.py)
venturi/
  cli.py                   command-line options, interactive mode, the six steps
  fluids.py                fluid table; gas density versus pressure
  config.py                all inputs, sizing, warnings
  geometry.py              wall profile R(z) with rounded corners
  fvmesh.py                cells, face areas, volumes
  projection.py            pressure correction that keeps mass exact
  solver/
    solver.py              the time-stepping loop
    kernels.py             the heavy loops, compiled with Numba
  turbulence.py            mixing length and Baldwin-Lomax models
  validation.py            checks, pressure taps, C_d
  optimizer.py             quick choice of D, beta, widening angle (mode 2)
  export/
    paraview.py            .vts and .stl
    cad.py                 .dxf and .step
    plots.py               .svg
    tables.py              .csv, .ply and the .md report
tests/                     automatic tests (section 13)
results/                   created by the runs, not stored in git
```

## 13. Tests

```bash
pytest tests/          # 53 tests, about 10 seconds
```

| file | what it checks |
|---|---|
| `test_conservation.py` | cell volumes add up to the tube volume; no fake mass in uniform flow; the correction removes all imbalance and leaves the inlet and wall untouched; the pressure matrix is symmetric; the same on a grid with shorter throat cells, whose spacing changes smoothly |
| `test_physics.py` | straight pipe against the exact laminar solution (pressure gradient and velocity profile); result independent of the time step; inlet profile shape and flow rate |
| `test_turbulence_models.py` | eddy viscosity never negative, larger than plain viscosity at high Re, zero for laminar; unknown model names are refused |
| `test_geometry.py` | radius and slope are continuous where the rounded corners join; slope matches a numerical derivative |
| `test_inputs.py` | wrong inputs are refused; sizing gives the requested pressure drop; gas density follows pressure; Mach and cavitation warnings; command-line options reach the program; the optimiser; taps read the wall at their exact position; the cavitation check |
| `test_export.py` | every file carries the right value on the right point; missing libraries skip files cleanly |

## 14. Questions and problems

**"Steady state reached" fails.** Raise `--max-iter`. If the residual
stops going down, try a lower `--cfl` (for example 0.4). On fine grids see
[section 11](#11-speed).

**"First cell y+" fails.** The wall cells are too thick for the
turbulence model. Raise `--clustering` (for example 3.0) or `--Nr`.

**The run is too slow.** Try a smaller grid first (for example
`--Nz 60 --Nr 24`) to check the setup, then go back to the default.

**My gas case warns about Mach.** The pressure drop is too large compared
with the inlet pressure. The results are then only indicative, because the
gas density changes along the tube and the program keeps it fixed.

**Where do I set a fluid that is not in the list?**
`--fluid custom --rho 850 --mu 0.003 --p-vap 500`

**Why is C_d not exactly 1?** Friction at the wall makes the real pressure
drop larger than the frictionless one, so the real flow for a given drop is
a bit smaller. C_d measures exactly that.

---

## A note on AI

Claude was used as a coding assistant for the solver design, the
diagnostics and this README. The numbers in section 10 come from running
the code, and the tests reproduce them.

## Licence

MIT, see [LICENSE](LICENSE).
