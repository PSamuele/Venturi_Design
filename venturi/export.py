import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless SVG generation
import matplotlib.pyplot as plt

try:
    import pyvista as pv
except ImportError:
    pv = None

try:
    import ezdxf
except ImportError:
    ezdxf = None

try:
    import cadquery as cq
except ImportError:
    cq = None

from .config import VenturiConfig
from .geometry import VenturiGeometry, profile_points, radius_at
from .compat import MeshView as VenturiMesh


def export_paraview_2d(mesh: VenturiMesh, uz: np.ndarray, ur: np.ndarray, p: np.ndarray, config: VenturiConfig) -> str:
    """Export the 2D grid as a .vts file for ParaView."""
    filepath = os.path.join(config.output_dir, "venturi_cfd_2d.vts")
    if pv is None:
        print("Warning: pyvista is not installed; cannot export vts.")
        return filepath

    # PyVista uses 3D coordinates. x=z, y=r, z=0
    x = mesh.z
    y = mesh.r
    z = np.zeros_like(x)
    grid = pv.StructuredGrid(x, y, z)

    vel_mag = np.sqrt(uz**2 + ur**2)
    velocity_vec = np.column_stack((uz.ravel(), ur.ravel(), np.zeros_like(uz.ravel())))

    grid.point_data['Velocity'] = velocity_vec
    grid.point_data['Velocity_Magnitude'] = vel_mag.ravel()
    grid.point_data['Pressure'] = p.ravel()
    grid.point_data['Axial_Velocity'] = uz.ravel()
    grid.point_data['Radial_Velocity'] = ur.ravel()

    grid.save(filepath)
    return filepath


def export_paraview_3d(mesh: VenturiMesh, uz: np.ndarray, ur: np.ndarray, p: np.ndarray, config: VenturiConfig) -> str:
    """Export the revolved 3D mesh as a .vtp file for ParaView."""
    filepath = os.path.join(config.output_dir, "venturi_cfd_3d.vtp")
    if pv is None:
        print("Warning: pyvista is not installed; cannot export vtp.")
        return filepath

    x = mesh.z
    y = mesh.r
    z = np.zeros_like(x)
    grid = pv.StructuredGrid(x, y, z)

    vel_mag = np.sqrt(uz**2 + ur**2)
    velocity_vec = np.column_stack((uz.ravel(), ur.ravel(), np.zeros_like(uz.ravel())))
    
    grid.point_data['Velocity'] = velocity_vec
    grid.point_data['Velocity_Magnitude'] = vel_mag.ravel()
    grid.point_data['Pressure'] = p.ravel()

    # Create 3D by revolving the 2D plane
    poly = grid.extract_surface(algorithm='dataset_surface')
    
    try:
        revolved = poly.revolve(angle=360, resolution=72)
    except AttributeError:
        # Fallback to extrude_rotate if revolve is not an attribute
        revolved = poly.extrude_rotate(resolution=72, angle=360.0, capping=False)

    revolved.save(filepath)
    return filepath


def export_dxf(geom: VenturiGeometry, config: VenturiConfig) -> str:
    """Export the 2D wall profile to DXF."""
    filepath = os.path.join(config.output_dir, "venturi_profile.dxf")
    if ezdxf is None:
        print("Warning: ezdxf is not installed; cannot export dxf.")
        return filepath

    doc = ezdxf.new('R2010')
    msp = doc.modelspace()

    doc.layers.add('PROFILE_INNER', color=1)
    doc.layers.add('PROFILE_OUTER', color=2)
    doc.layers.add('CENTERLINE', color=3, linetype='CENTER')
    doc.layers.add('DIMENSIONS', color=4)

    z_arr, r_arr = profile_points(geom, n_points=200)

    # Inner profile (wetted surface)
    points_inner = [(z, r) for z, r in zip(z_arr, r_arr)]
    msp.add_lwpolyline(points_inner, dxfattribs={'layer': 'PROFILE_INNER'})

    # Outer profile (mirrored)
    points_outer = [(z, -r) for z, r in zip(z_arr, r_arr)]
    msp.add_lwpolyline(points_outer, dxfattribs={'layer': 'PROFILE_OUTER'})

    # Centerline
    msp.add_line((0, 0), (geom.L_total, 0), dxfattribs={'layer': 'CENTERLINE'})

    doc.saveas(filepath)
    return filepath


def export_stl(geom: VenturiGeometry, config: VenturiConfig) -> str:
    """Export the 3D Venturi surface to STL."""
    filepath = os.path.join(config.output_dir, "venturi_3d.stl")
    if pv is None:
        print("Warning: pyvista is not installed; cannot export stl.")
        return filepath

    z_arr, r_arr = profile_points(geom, n_points=200)
    points = np.column_stack((z_arr, r_arr, np.zeros_like(z_arr)))
    
    # Create line from points
    lines = np.empty((len(points) - 1, 3), dtype=int)
    lines[:, 0] = 2
    lines[:, 1] = np.arange(0, len(points) - 1)
    lines[:, 2] = np.arange(1, len(points))
    
    line_poly = pv.PolyData(points, lines=lines)
    
    try:
        surf = line_poly.revolve(angle=360, resolution=72)
    except AttributeError:
        surf = line_poly.extrude_rotate(resolution=72, angle=360.0, capping=False)

    surf.save(filepath)
    return filepath


def export_step(geom: VenturiGeometry, config: VenturiConfig) -> str | None:
    """Export the 3D solid model to STEP."""
    if cq is None:
        print("Warning: cadquery is not installed; skipping STEP export.")
        return None

    filepath = os.path.join(config.output_dir, "venturi_3d.step")
    
    z_arr, r_arr = profile_points(geom, n_points=200)
    
    # Create closed profile for a solid revolution
    pts = [(z, r) for z, r in zip(z_arr, r_arr)]
    pts = [(0.0, 0.0)] + pts + [(geom.L_total, 0.0)]
    
    try:
        # Revolve around X axis (Z is default for cadquery, so we specify axis)
        wp = cq.Workplane("XY").polyline(pts).close().revolve(360, (0, 0, 0), (1, 0, 0))
        wp.val().exportStep(filepath)
        return filepath
    except Exception as e:
        print(f"STEP export failed: {e}")
        return None


def export_svg_geometry(geom: VenturiGeometry, config: VenturiConfig) -> str:
    """Export the geometric engineering drawing to SVG."""
    filepath = os.path.join(config.output_dir, "venturi_geometry.svg")
    
    z_arr, r_arr = profile_points(geom, n_points=500)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Profiles
    ax.plot(z_arr, r_arr, 'b-', label='Profilo Superiore', linewidth=2)
    ax.plot(z_arr, -r_arr, 'b-', label='Profilo Inferiore', linewidth=2)
    
    # Centerline
    ax.plot([0, geom.L_total], [0, 0], 'k--', alpha=0.5, label='Asse')
    
    # Annotations
    ax.set_title(f"Venturi geometry - $D_{{in}}$: {config.D_inlet*1e3:.1f} mm, "
                 f"$d_{{th}}$: {config.d_throat*1e3:.1f} mm\n"
                 f"$\\alpha_{{conv}}$: {config.alpha_conv_deg}°, $\\alpha_{{div}}$: {config.alpha_div_deg}°")
    ax.set_xlabel("Asse Z (m)")
    ax.set_ylabel("Radius R (m)")
    ax.grid(True, linestyle=':', alpha=0.6)
    # Rimosso ax.axis('equal') per evitare linee lunghissime e sottili
    
    plt.tight_layout()
    fig.savefig(filepath, format='svg')
    plt.close(fig)
    return filepath


def export_svg_results(mesh: VenturiMesh, uz: np.ndarray, ur: np.ndarray, p: np.ndarray,
                       geom: VenturiGeometry, config: VenturiConfig) -> str:
    """Export the CFD results as a vector SVG figure."""
    filepath = os.path.join(config.output_dir, "venturi_results.svg")
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=False)
    
    vel_mag = np.sqrt(uz**2 + ur**2)
    z = mesh.z
    r = mesh.r

    # 1. Velocity magnitude + quiver arrows
    c1 = ax1.contourf(z, r, vel_mag, levels=50, cmap='viridis')
    fig.colorbar(c1, ax=ax1, label='Velocity (m/s)')
    ax1.set_title("Velocity Field")
    ax1.set_ylabel("Radius (m)")
    
    # Quiver arrows for flow direction
    skip = max(1, z.shape[0] // 30)
    skip_r = max(1, z.shape[1] // 15)
    ax1.quiver(z[::skip, ::skip_r], r[::skip, ::skip_r], 
               uz[::skip, ::skip_r], ur[::skip, ::skip_r], 
               color='white', alpha=0.7)
    # Rimosso ax1.set_aspect('equal') per evitare grafici illeggibili

    # 2. Pressure contours
    c2 = ax2.contourf(z, r, p, levels=50, cmap='inferno')
    fig.colorbar(c2, ax=ax2, label='Pressure (Pa)')
    ax2.set_title("Pressure Field")
    ax2.set_ylabel("Radius (m)")
    # Rimosso ax2.set_aspect('equal') per evitare grafici illeggibili
    
    # 3. Centerline pressure profile with Bernoulli overlay
    z_cl = z[:, 0]
    p_cl = p[:, 0]
    
    # Bernoulli analytical estimate: v(z) ≈ v_in * (A_in / A(z))
    v_in = config.v_inlet
    r_wall_cl = radius_at(z_cl, geom)
    # Avoid division by zero at the wall radius
    r_wall_cl = np.maximum(r_wall_cl, 1e-10)
    v_cl_analytical = v_in * (config.R_inlet / r_wall_cl) ** 2
    p_analytical = config.p_inlet + 0.5 * config.rho * (v_in**2 - v_cl_analytical**2)
    
    ax3.plot(z_cl, p_cl, 'r-', linewidth=2, label='CFD')
    ax3.plot(z_cl, p_analytical, 'k--', alpha=0.7, label='Bernoulli (ideal)')
    ax3.set_title("Centreline Pressure Profile")
    ax3.set_xlabel("Axial position z (m)")
    ax3.set_ylabel("Pressure (Pa)")
    ax3.legend()
    ax3.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    fig.savefig(filepath, format='svg', bbox_inches='tight')
    plt.close(fig)
    return filepath


def export_pointcloud(mesh: VenturiMesh, uz: np.ndarray, ur: np.ndarray, p: np.ndarray, config: VenturiConfig) -> tuple[str, str]:
    """Export a 3D point cloud of the results to PLY and CSV."""
    ply_path = os.path.join(config.output_dir, "venturi_pointcloud.ply")
    csv_path = os.path.join(config.output_dir, "venturi_pointcloud.csv")
    
    angles = np.radians(np.arange(0, 360, 45))
    vel_mag = np.sqrt(uz**2 + ur**2)
    
    points_3d = []
    p_data = []
    v_data = []
    
    # Flatten 2D arrays
    z_flat = mesh.z.ravel()
    r_flat = mesh.r.ravel()
    p_flat = p.ravel()
    v_flat = vel_mag.ravel()
    
    for theta in angles:
        x_rot = z_flat
        y_rot = r_flat * np.cos(theta)
        z_rot = r_flat * np.sin(theta)
        
        points_3d.append(np.column_stack((x_rot, y_rot, z_rot)))
        p_data.append(p_flat)
        v_data.append(v_flat)
        
    points_3d = np.vstack(points_3d)
    p_data = np.concatenate(p_data)
    v_data = np.concatenate(v_data)
    
    # Write CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'y', 'z', 'pressure', 'velocity_magnitude'])
        for i in range(len(points_3d)):
            writer.writerow([points_3d[i,0], points_3d[i,1], points_3d[i,2], p_data[i], v_data[i]])
            
    # Write PLY using pyvista if available
    if pv is not None:
        pc = pv.PolyData(points_3d)
        pc.point_data['Pressure'] = p_data
        pc.point_data['Velocity_Magnitude'] = v_data
        
        # Save as PLY (Note: pyvista ply writer does not always save point arrays. vtp is better, but user requested PLY)
        try:
            pc.save(ply_path)
        except Exception as e:
            print(f"PLY export failed: {e}")
            ply_path = ""
    else:
        ply_path = ""

    return ply_path, csv_path


def export_report(config: VenturiConfig, geom: VenturiGeometry, validation_results: dict | None = None) -> str:
    """Export a validation report in Markdown."""
    filepath = os.path.join(config.output_dir, "venturi_report.md")
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("# Venturi Tube Design Report\n\n")
        
        f.write("## 1. Configuration Parameters\n")
        f.write(f"- Fluid: `{config.fluid_name}`\n")
        f.write(f"- Density: `{config.rho:.2f}` kg/m³\n")
        f.write(f"- Dynamic viscosity: `{config.mu:.2e}` Pa·s\n")
        f.write(f"- Volumetric flow rate (Q): `{config.Q:.4f}` m³/s\n\n")
        
        f.write("## 2. Geometry\n")
        f.write(f"- Inlet diameter ($D_{{in}}$): `{config.D_inlet*1e3:.2f}` mm\n")
        f.write(f"- Throat diameter ($d_{{th}}$): `{config.d_throat*1e3:.2f}` mm\n")
        f.write(f"- Ratio $\\beta$: `{config.beta:.3f}`\n")
        f.write(f"- Convergent angle: `{config.alpha_conv_deg}`°\n")
        f.write(f"- Divergent angle: `{config.alpha_div_deg}`°\n")
        f.write(f"- Total length: `{geom.L_total*1e3:.2f}` mm\n\n")
        
        if validation_results:
            f.write("## 3. Validation Results\n")
            f.write("| Metric | Value | Status |\n")
            f.write("|---------|--------|--------|\n")
            for k, v in validation_results.items():
                if isinstance(v, dict) and 'passed' in v:
                    status = "✅ PASS" if v['passed'] else "❌ FAIL"
                    val = v.get('value', 'N/A')
                    if isinstance(val, float):
                        f.write(f"| {k} | {val:.4f} | {status} |\n")
                    else:
                        f.write(f"| {k} | {val} | {status} |\n")
                else:
                    f.write(f"| {k} | {v} | - |\n")
            f.write("\n")
            
        f.write("## 4. Generated Files\n")
        f.write("All outputs are written to the configured folder.\n")
        
    return filepath


def export_all(mesh: VenturiMesh, uz: np.ndarray, ur: np.ndarray, p: np.ndarray, geom: VenturiGeometry, config: VenturiConfig, validation_results=None) -> list[str]:
    """Run every available export."""
    os.makedirs(config.output_dir, exist_ok=True)
    files = []
    
    print("[6/6] Writing output files...")
    
    f = export_paraview_2d(mesh, uz, ur, p, config)
    if f: files.append(f); print(f"      OK {f}")
        
    f = export_paraview_3d(mesh, uz, ur, p, config)
    if f: files.append(f); print(f"      OK {f}")
        
    f = export_dxf(geom, config)
    if f: files.append(f); print(f"      OK {f}")
        
    f = export_stl(geom, config)
    if f: files.append(f); print(f"      OK {f}")
        
    f = export_step(geom, config)
    if f: files.append(f); print(f"      OK {f}")
        
    f = export_svg_geometry(geom, config)
    if f: files.append(f); print(f"      OK {f}")
        
    f = export_svg_results(mesh, uz, ur, p, geom, config)
    if f: files.append(f); print(f"      OK {f}")
        
    ply_f, csv_f = export_pointcloud(mesh, uz, ur, p, config)
    if ply_f: files.append(ply_f); print(f"      OK {ply_f}")
    if csv_f: files.append(csv_f); print(f"      OK {csv_f}")
        
    f = export_report(config, geom, validation_results)
    if f: files.append(f); print(f"      OK {f}")
        
    print("Export completed successfully.")
    return files
