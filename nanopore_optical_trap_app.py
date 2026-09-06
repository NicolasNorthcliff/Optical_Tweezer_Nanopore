"""Interactive nanopore + optical-trap trajectory simulator.

Run:
    pip install streamlit numpy pandas plotly matplotlib pillow
    streamlit run nanopore_optical_trap_app.py

This quick analytic model is intended for qualitative exploration. Replace
the analytic electric/flow fields with COMSOL field maps for quantitative use.
"""
from __future__ import annotations

from io import BytesIO
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from PIL import Image
import streamlit as st

st.set_page_config(page_title="Nanopore × Optical Trap", page_icon="🔬", layout="wide")


def particle_shape(radius_um: float, pore_um: float, rho_um: float, z_um: float,
                   deformable: bool, max_strain: float) -> tuple[float, float, float, bool]:
    """Return transverse radius, axial radius, radial strain, and passage feasibility.

    This is a qualitative, volume-conserving oblate/prolate deformation model.
    Near the pore, the particle contracts in x-y and elongates in z. It may pass
    only if the radial strain required by the available aperture is no greater
    than the user-selected maximum strain.
    """
    available_radius = max(0.0, pore_um - rho_um)
    required_strain = max(0.0, 1.0 - available_radius / max(radius_um, 1e-12))
    can_squeeze = required_strain <= max_strain + 1e-12
    if not deformable or required_strain <= 0 or not can_squeeze:
        return radius_um, radius_um, 0.0, required_strain <= 0

    rigid_contact_z = math.sqrt(max(0.0, radius_um**2 - available_radius**2))
    distance_from_contact = max(0.0, abs(z_um) - rigid_contact_z)
    engagement = max(0.0, 1.0 - distance_from_contact / max(radius_um, 1e-12))
    strain = required_strain * engagement
    transverse = radius_um * (1.0 - strain)
    # Approximate volume conservation: r_xy^2 * r_z = r_0^3.
    axial = radius_um**3 / max(transverse**2, 1e-12)
    return transverse, axial, strain, can_squeeze


def simulate(cfg: dict, axes: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(int(cfg["seed"]))
    n_steps = min(2500, max(200, int(cfg["duration_ms"] / cfg["dt_ms"])))
    dt = cfg["duration_ms"] / n_steps
    radius_um = cfg["radius_nm"] / 1000
    pore_um = cfg["pore_radius_nm"] / 1000
    gamma = max(0.05, 6 * np.pi * cfg["viscosity_mpas"] * radius_um)
    contrast = max(0.05, (cfg["particle_n"]**2 - cfg["medium_n"]**2) /
                   (cfg["particle_n"]**2 + 2 * cfg["medium_n"]**2))
    pos = np.array(cfg["initial_position_um"], dtype=float)
    rows = []
    axis_i = {"x": 0, "y": 1, "z": 2}
    f_gravity = np.zeros(3)
    pore_clearance_um = pore_um - radius_um
    max_strain = cfg["max_radial_strain_pct"] / 100.0
    if cfg["gravity_enabled"]:
        direction = cfg["gravity_direction"]
        sign = 1.0 if direction[0] == "+" else -1.0
        idx = axis_i[direction[1]]
        volume_m3 = 4 / 3 * np.pi * (radius_um * 1e-6)**3
        # Effective weight includes buoyancy: (rho_particle-rho_medium) V g.
        effective_weight_pn = ((cfg["density"] - cfg["medium_density"]) *
                               volume_m3 * 9.80665 * 1e12)
        f_gravity[idx] = sign * effective_weight_pn

    for i in range(n_steps + 1):
        x, y, z = pos
        rho2 = x*x + y*y
        distance = np.linalg.norm(pos)
        envelope = np.exp(-rho2 / (2 * max(0.12, pore_um)**2)) * np.exp(-max(0, z) / 1.9)

        # Effective combined EP/DEP/EOF/hydrodynamic nanopore attraction.
        pore_strength = cfg["pore_force_scale"] * (0.018 * cfg["voltage_mv"] + 0.004 * cfg["pressure_mbar"]) * envelope
        f_pore = np.array([
            -pore_strength * x / max(0.18, pore_um),
            -pore_strength * y / max(0.18, pore_um),
            -pore_strength * (0.55 + z / 3),
        ])

        # Harmonic optical-gradient-force proxy around each beam's focal point.
        f_opt = np.zeros(3)
        for axis in axes:
            idx = axis_i[axis]
            power = cfg[f"power_{axis}_mw"]
            waist = cfg[f"waist_{axis}_um"]
            focus = np.asarray(cfg[f"focus_{axis}_um"])
            stiffness = cfg["optical_force_scale"] * 0.035 * power * contrast / max(0.2, waist**2)
            # A real focused beam confines transversely and more weakly axially.
            delta = pos - focus
            transverse = [j for j in range(3) if j != idx]
            f_opt[transverse] += -stiffness * delta[transverse]
            f_opt[idx] += -0.25 * stiffness * delta[idx]

        surface_gap = max(0.015, distance - radius_um - pore_um)
        f_vdw = min(0.9, cfg["hamaker_1e21j"] * 0.0009 / surface_gap**2)
        f_det = f_pore + f_opt + np.array([0.0, 0.0, -f_vdw]) + f_gravity

        sigma = math.sqrt(0.016 * cfg["temperature_k"] / 298 * dt / gamma) if cfg["brownian"] else 0
        brownian_step = rng.normal(0, sigma, 3)
        displacement = (f_det / gamma) * dt + brownian_step
        # Spatial sub-step limiter: prevents a large force or Brownian kick from
        # teleporting the particle across the membrane in one animation frame.
        max_displacement = 0.12 * max(0.02, min(radius_um, pore_um))
        displacement_norm = np.linalg.norm(displacement)
        if displacement_norm > max_displacement:
            displacement *= max_displacement / displacement_norm
        proposed_pos = pos + displacement

        proposed_rho = math.hypot(proposed_pos[0], proposed_pos[1])
        transverse_r, axial_r, strain, can_squeeze = particle_shape(
            radius_um, pore_um, proposed_rho, proposed_pos[2],
            cfg["deformable_particle"], max_strain)
        rim_distance = pore_um - proposed_rho
        if proposed_rho >= pore_um:
            minimum_center_z = axial_r
            blocked_by_solid = True
        elif rim_distance + 1e-12 < transverse_r:
            minimum_center_z = axial_r * math.sqrt(
                max(0.0, 1.0 - (rim_distance / max(transverse_r, 1e-12))**2))
            blocked_by_solid = True
        else:
            minimum_center_z = 0.0
            blocked_by_solid = False

        collided = False
        # The simulated particle starts above the membrane. Clamp every trial
        # step that would overlap the solid membrane, preventing time-step
        # tunnelling even when the deterministic or Brownian step is large.
        if blocked_by_solid and pos[2] >= 0.0 and proposed_pos[2] < minimum_center_z:
            proposed_pos[2] = minimum_center_z
            collided = True
        pos = proposed_pos
        final_rho = math.hypot(pos[0], pos[1])
        transverse_r, axial_r, strain, can_squeeze = particle_shape(
            radius_um, pore_um, final_rho, pos[2],
            cfg["deformable_particle"], max_strain)

        rows.append({
            "t_ms": i * dt, "x_um": pos[0], "y_um": pos[1], "z_um": pos[2],
            "F_opt_x_pN": f_opt[0], "F_opt_y_pN": f_opt[1], "F_opt_z_pN": f_opt[2],
            "F_pore_x_pN": f_pore[0], "F_pore_y_pN": f_pore[1], "F_pore_z_pN": f_pore[2],
            "F_VDW_z_pN": -f_vdw,
            "F_gravity_x_pN": f_gravity[0], "F_gravity_y_pN": f_gravity[1],
            "F_gravity_z_pN": f_gravity[2], "F_gravity_mag_pN": np.linalg.norm(f_gravity),
            "F_total_x_pN": f_det[0], "F_total_y_pN": f_det[1], "F_total_z_pN": f_det[2],
            "F_opt_mag_pN": np.linalg.norm(f_opt),
            "F_pore_mag_pN": np.linalg.norm(f_pore), "F_VDW_mag_pN": f_vdw,
            "F_total_mag_pN": np.linalg.norm(f_det),
            "particle_fits_pore": pore_clearance_um > 0,
            "pore_clearance_um": pore_clearance_um,
            "membrane_collision": collided,
            "particle_transverse_radius_um": transverse_r,
            "particle_axial_radius_um": axial_r,
            "particle_radial_strain_pct": 100.0 * strain,
            "deformation_allows_passage": can_squeeze,
        })
    return pd.DataFrame(rows)


def trajectory_figure(df: pd.DataFrame, pore_radius_um: float) -> go.Figure:
    th = np.linspace(0, 2*np.pi, 90)
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=df.x_um, y=df.y_um, z=df.z_um, mode="lines",
                               name="Trajectory", line=dict(color="#45dfcb", width=5)))
    last = df.iloc[-1]
    u = np.linspace(0, 2*np.pi, 36)
    v = np.linspace(0, np.pi, 24)
    uu, vv = np.meshgrid(u, v)
    rt, rz = last.particle_transverse_radius_um, last.particle_axial_radius_um
    sx = last.x_um + rt * np.cos(uu) * np.sin(vv)
    sy = last.y_um + rt * np.sin(uu) * np.sin(vv)
    sz = last.z_um + rz * np.cos(vv)
    fig.add_trace(go.Surface(x=sx, y=sy, z=sz, name="Particle", showscale=False,
                             colorscale=[[0, "#75d8ce"], [1, "#eafffb"]], opacity=.95,
                             hovertemplate="Deformed particle<extra></extra>"))
    fig.add_trace(go.Scatter3d(x=pore_radius_um*np.cos(th), y=pore_radius_um*np.sin(th),
                               z=np.zeros_like(th), mode="lines", name="Pore rim",
                               line=dict(color="#ff6b8a", width=5)))
    white_axis = dict(
        title_font=dict(color="#ffffff", size=15),
        tickfont=dict(color="#ffffff", size=12),
        color="#ffffff",
        gridcolor="#3b4966",
        zerolinecolor="#ffffff",
    )
    fig.update_layout(
        height=570, margin=dict(l=0, r=0, t=25, b=0), template="plotly_dark",
        font=dict(color="#ffffff", size=13),
        scene=dict(
            xaxis=dict(title="x (µm)", **white_axis),
            yaxis=dict(title="y (µm)", **white_axis),
            zaxis=dict(title="z (µm)", **white_axis),
            bgcolor="#080d18", aspectmode="cube",
        ),
        paper_bgcolor="#080d18",
        legend=dict(orientation="h", font=dict(color="#ffffff")),
    )
    return fig


def geometry_figure(cfg: dict, axes: list[str], view: str) -> go.Figure:
    """Top (x-y) or cross-section (x-z) schematic in physical coordinates."""
    ia, ib = ((0, 1) if view == "top" else (0, 2))
    labels = (("x (µm)", "y (µm)") if view == "top" else ("x (µm)", "z (µm)"))
    colors = {"x": "#ff6b8a", "y": "#45dfcb", "z": "#73a7ff"}
    fig = go.Figure()
    pore = cfg["pore_radius_nm"] / 1000
    th = np.linspace(0, 2*np.pi, 120)
    if view == "top":
        fig.add_scatter(x=pore*np.cos(th), y=pore*np.sin(th), mode="lines",
                        name="Nanopore at (0,0,0)", line=dict(color="#f4bd62", width=4), fill="toself",
                        fillcolor="rgba(244,189,98,.08)")
    else:
        fig.add_shape(type="rect", x0=-2.6, x1=2.6, y0=-.10, y1=.10,
                      line=dict(color="#f4bd62"), fillcolor="rgba(244,189,98,.18)")
        fig.add_shape(type="rect", x0=-pore, x1=pore, y0=-.12, y1=.12,
                      line=dict(color="#080d18"), fillcolor="#080d18")

    for axis in axes:
        focus = np.asarray(cfg[f"focus_{axis}_um"])
        waist = cfg[f"waist_{axis}_um"]
        power = cfg[f"power_{axis}_mw"]
        a, b = focus[ia], focus[ib]
        if (view == "top" and axis in ("x", "y")) or (view == "cross" and axis in ("x", "z")):
            horizontal = axis == "x"
            fig.add_shape(type="rect", x0=(-2.6 if horizontal else a-waist), x1=(2.6 if horizontal else a+waist),
                          y0=(b-waist if horizontal else -1.0), y1=(b+waist if horizontal else 3.1),
                          line=dict(color=colors[axis], width=1), fillcolor=colors[axis], opacity=.10)
            fig.add_annotation(x=a, y=b, text=f"{axis.upper()} beam<br>{power:g} mW · w₀={waist:g} µm",
                               showarrow=True, arrowcolor=colors[axis], font=dict(color=colors[axis], size=11))
        else:
            # Beam normal to the displayed plane appears as its waist circle.
            fig.add_shape(type="circle", x0=a-waist, x1=a+waist, y0=b-waist, y1=b+waist,
                          line=dict(color=colors[axis], width=2), fillcolor=colors[axis], opacity=.14)
            fig.add_annotation(x=a, y=b, text=f"{axis.upper()} beam ⊙<br>{power:g} mW · w₀={waist:g} µm",
                               showarrow=True, arrowcolor=colors[axis], font=dict(color=colors[axis], size=11))

    start = np.asarray(cfg["initial_position_um"])
    radius = cfg["radius_nm"] / 1000
    fig.add_shape(type="circle", x0=start[ia]-radius, x1=start[ia]+radius,
                  y0=start[ib]-radius, y1=start[ib]+radius,
                  line=dict(color="#45dfcb", width=3), fillcolor="rgba(234,255,251,.75)")
    fig.add_scatter(x=[start[ia]], y=[start[ib]], mode="markers+text", name="Initial particle",
                    marker=dict(size=4, color="white"),
                    text=[f"start ({start[0]:g}, {start[1]:g}, {start[2]:g}) µm"], textposition="top right")
    fig.add_scatter(x=[0], y=[0], mode="markers+text", name="Origin",
                    marker=dict(size=7, symbol="x", color="#f4bd62"), text=["pore center (0,0,0)"],
                    textposition="bottom right")
    fig.update_layout(height=430, template="plotly_dark", margin=dict(l=10, r=10, t=45, b=15),
                      title=dict(text="Top view (x–y)" if view == "top" else "Cross-section (x–z)", font=dict(color="#ffffff")),
                      font=dict(color="#ffffff", size=12),
                      xaxis=dict(title=labels[0], range=[-2.7, 2.7], zeroline=True, gridcolor="#26324c", color="#ffffff"),
                      yaxis=dict(title=labels[1], range=[-2.7, 2.7] if view == "top" else [-1.1, 3.2],
                                 scaleanchor="x", scaleratio=1, gridcolor="#26324c", color="#ffffff"),
                      paper_bgcolor="#080d18", plot_bgcolor="#080d18", showlegend=False)
    return fig


def force_figure(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.18,
                        subplot_titles=("Signed net-force components", "Magnitude by physical source"))
    component_colors = {"x": "#ffcf5c", "y": "#45dfcb", "z": "#a68cff"}
    for axis in "xyz":
        fig.add_trace(go.Scatter(x=df.t_ms, y=df[f"F_total_{axis}_pN"],
                                 name=f"Net F{axis}", line=dict(color=component_colors[axis], width=2.5)), row=1, col=1)
    sources = [("F_opt_mag_pN", "Optical |F|", "#73a7ff"),
               ("F_pore_mag_pN", "EP/DEP/EOF/pore |F|", "#ff6b8a"),
               ("F_VDW_mag_pN", "VDW |F|", "#ff9f43"),
               ("F_gravity_mag_pN", "Effective gravity |F|", "#70e000"),
               ("F_total_mag_pN", "Total |F|", "#ffffff")]
    for key, name, color in sources:
        fig.add_trace(go.Scatter(x=df.t_ms, y=df[key], name=name,
                                 line=dict(color=color, width=2.7, dash="dot" if key == "F_total_mag_pN" else "solid")), row=2, col=1)
    fig.update_layout(height=700, template="plotly_dark", margin=dict(l=45, r=20, t=125, b=40),
                      paper_bgcolor="#080d18", plot_bgcolor="#080d18",
                      font=dict(size=14, color="#ffffff"),
                      legend=dict(orientation="h", x=0, xanchor="left", y=1.22, yanchor="top",
                                  font=dict(color="#ffffff", size=11),
                                  bgcolor="rgba(8,13,24,.92)", bordercolor="#3b4966", borderwidth=1))
    fig.update_annotations(font=dict(color="#ffffff", size=15))
    fig.update_xaxes(title_text="Time (ms)", gridcolor="#3b4966", color="#ffffff", row=2, col=1)
    fig.update_yaxes(title_text="Force (pN)", gridcolor="#3b4966", zerolinecolor="#ffffff", color="#ffffff", row=1, col=1)
    fig.update_yaxes(title_text="Magnitude (pN)", gridcolor="#3b4966", color="#ffffff", row=2, col=1)
    return fig


def analyze_hovering(df: pd.DataFrame, particle_radius_um: float) -> dict:
    """Detect Brownian fluctuations around a stable force-balance height."""
    n_tail = max(40, len(df) // 4)
    tail = df.tail(n_tail)
    half = max(1, n_tail // 2)
    z_mean = float(tail.z_um.mean())
    z_std = float(tail.z_um.std(ddof=0))
    z_drift = abs(float(tail.z_um.iloc[:half].mean() - tail.z_um.iloc[half:].mean()))
    mean_fz = float(tail.F_total_z_pN.mean())
    force_scale = max(0.02, float(tail.F_total_z_pN.abs().median()))
    collision_fraction = float(tail.membrane_collision.mean())

    if z_std > 1e-5:
        restoring_slope = float(np.polyfit(tail.z_um, tail.F_total_z_pN, 1)[0])
    else:
        restoring_slope = -np.inf

    above_membrane = z_mean > float(tail.particle_axial_radius_um.mean()) + 0.03
    low_drift = z_drift < max(0.05, 0.30 * particle_radius_um)
    localized = z_std < max(0.20, 0.75 * particle_radius_um)
    near_force_balance = abs(mean_fz) < max(0.05, 0.25 * force_scale)
    hovering = (above_membrane and low_drift and localized and near_force_balance and
                restoring_slope < 0 and collision_fraction < 0.05 and
                not bool((tail.z_um < 0).any()))
    return dict(hovering=hovering, z_mean_um=z_mean, z_std_um=z_std,
                z_drift_um=z_drift, mean_fz_pn=mean_fz,
                restoring_slope_pn_per_um=restoring_slope)


def make_gif(df: pd.DataFrame, pore_radius_um: float, view: str = "top") -> bytes:
    """Render either an x-y top view or an x-z cross-section trajectory GIF."""
    sample = np.linspace(1, len(df), 60, dtype=int)
    if view == "top":
        horizontal, vertical = "x_um", "y_um"
        x_label, y_label = "x (µm)", "y (µm)"
    else:
        horizontal, vertical = "x_um", "z_um"
        x_label, y_label = "x (µm)", "z (µm)"
    max_rt = float(df.particle_transverse_radius_um.max())
    max_rz = float(df.particle_axial_radius_um.max())
    x_lim = max(2.2, (np.abs(df[horizontal]).max() + max_rt) * 1.12)
    if view == "top":
        y_limits = (-x_lim, x_lim)
    else:
        z_min = min(-0.55, float(df.z_um.min()) - max_rz - 0.15)
        z_max = max(2.2, float(df.z_um.max()) + max_rz + 0.15)
        y_limits = (z_min, z_max)
    frames = []
    for end in sample:
        fig, ax = plt.subplots(figsize=(5, 5), dpi=90)
        fig.patch.set_facecolor("#080d18"); ax.set_facecolor("#080d18")
        ax.plot(df[horizontal].iloc[:end], df[vertical].iloc[:end], color="#45dfcb", lw=1.8)
        row = df.iloc[end-1]
        rt, rz = row.particle_transverse_radius_um, row.particle_axial_radius_um
        particle_width = 2 * rt
        particle_height = 2 * (rt if view == "top" else rz)
        ax.add_patch(Ellipse((row[horizontal], row[vertical]), particle_width, particle_height,
                             facecolor="#eafffb", edgecolor="#45dfcb", lw=2.2,
                             alpha=.95, zorder=5))
        if view == "top":
            ax.add_patch(plt.Circle((0, 0), pore_radius_um, fill=False,
                                    color="#ff6b8a", lw=2))
        else:
            # Membrane at z=0, drawn as two segments with the nanopore opening between them.
            ax.plot([-x_lim, -pore_radius_um], [0, 0], color="#f4bd62", lw=7,
                    solid_capstyle="butt")
            ax.plot([pore_radius_um, x_lim], [0, 0], color="#f4bd62", lw=7,
                    solid_capstyle="butt")
            ax.plot([-pore_radius_um, pore_radius_um], [0, 0], color="#ff6b8a",
                    lw=2, ls="--")
        ax.set(xlim=(-x_lim, x_lim), ylim=y_limits, xlabel=x_label, ylabel=y_label)
        ax.set_aspect("equal", adjustable="box"); ax.grid(alpha=.22, color="#64708a")
        ax.tick_params(colors="#ffffff")
        ax.xaxis.label.set_color("#ffffff"); ax.yaxis.label.set_color("#ffffff")
        ax.set_title("Top view (x–y)" if view == "top" else "Cross-section (x–z)",
                     color="#ffffff")
        buf = BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight"); plt.close(fig)
        frames.append(Image.open(buf).convert("P", palette=Image.Palette.ADAPTIVE))
    out = BytesIO(); frames[0].save(out, format="GIF", save_all=True, append_images=frames[1:],
                                   duration=70, loop=0, optimize=True)
    return out.getvalue()



# ---------------------------------------------------------------------------
# Precision / repeatability experiment models
# ---------------------------------------------------------------------------

def drag_pn_ms_per_nm(radius_nm: float, viscosity_mpas: float,
                      wall_gap_nm: float | None = None,
                      wall_correction: bool = True) -> float:
    """Stokes drag in pN*ms/nm, with an optional simple near-wall correction.

    Conversion is exact for the bulk Stokes term:
        gamma = 6*pi*eta*a
    where eta is converted from mPa*s to Pa*s and a from nm to m.

    The wall factor is a deliberately conservative phenomenological correction
    used only for sensitivity analysis. It is NOT a substitute for Faxen/Brenner
    hydrodynamics or a COMSOL solution near a nanopore.
    """
    eta = viscosity_mpas * 1e-3
    a_m = radius_nm * 1e-9
    gamma_si = 6.0 * np.pi * eta * a_m  # N*s/m
    # 1 (N*s/m) = 1e6 (pN*ms/nm)
    gamma = gamma_si * 1e6              # pN*ms/nm
    if wall_correction and wall_gap_nm is not None:
        gap = max(float(wall_gap_nm), 2.0)
        # Smooth conservative multiplier that becomes important within ~1 radius.
        gamma *= 1.0 + 0.55 * radius_nm / gap
    return gamma


def pore_force_z_gap_pn(gap_nm: float, cfg: dict) -> float:
    """Map the existing qualitative pore-force model onto a vertical surface gap.

    Negative is toward the membrane/pore. The particle is assumed centered over
    the pore. This keeps the new experiment tabs consistent with the original
    simulator while making clear that this remains a calibration proxy.
    """
    z_um = (cfg["radius_nm"] + max(gap_nm, 0.0)) / 1000.0
    pore_um = cfg["pore_radius_nm"] / 1000.0
    envelope = np.exp(-max(0.0, z_um) / 1.9)
    strength = (cfg["pore_force_scale"] *
                (0.018 * cfg["voltage_mv"] + 0.004 * cfg["pressure_mbar"]) *
                envelope)
    return -strength * (0.55 + z_um / 3.0)


def simulate_step_approach(cfg: dict, exp: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Repeated step-and-hold approach experiment.

    The commanded quantity is particle-surface gap, not bead-center z.
    At each target gap the trap focus is positioned accordingly, but the actual
    bead position is allowed to fluctuate and shift from the focus because of
    external pore force, Brownian motion, stage/focus jitter, detector noise,
    stiffness calibration error, and run-to-run drift.
    """
    rng = np.random.default_rng(int(exp["precision_seed"]))
    targets = np.arange(exp["start_gap_nm"],
                        exp["end_gap_nm"] - 0.5 * exp["step_nm"],
                        -exp["step_nm"])
    if len(targets) == 0:
        targets = np.array([exp["start_gap_nm"]], dtype=float)

    radius_nm = cfg["radius_nm"]
    dt_ms = exp["precision_dt_ms"]
    n_hold = max(40, int(exp["dwell_ms"] / dt_ms))
    repeats = int(exp["repeats"])
    k_nom = max(exp["kz_pn_per_nm"], 1e-6)

    all_rows = []
    for rep in range(repeats):
        # One slow offset per repeat represents refocus / stage re-zero error.
        repeat_offset = rng.normal(0.0, exp["repeatability_sigma_nm"])
        # Stiffness calibration changes from repeat to repeat.
        k_actual = k_nom * (1.0 + rng.normal(0.0, exp["stiffness_cv_pct"] / 100.0))
        k_actual = max(k_actual, 1e-6)

        # Start near the first commanded point.
        gap_true = float(targets[0] + repeat_offset)
        for j, target in enumerate(targets):
            # Commanded trap focus includes finite positioning resolution/jitter.
            focus_error = rng.normal(0.0, exp["command_sigma_nm"])
            focus_gap = float(target + repeat_offset + focus_error)

            # Simulate hold at each step.
            for q in range(n_hold):
                global_t_ms = (j * n_hold + q) * dt_ms
                # Slow drift specified in nm/min, modeled as random-walk-like
                # uncertainty over elapsed time.
                elapsed_min = max(global_t_ms / 60000.0, 0.0)
                drift_sigma = exp["drift_nm_per_min"] * np.sqrt(elapsed_min + 1e-12)
                drift = rng.normal(0.0, drift_sigma)

                gamma = drag_pn_ms_per_nm(
                    radius_nm, cfg["viscosity_mpas"], gap_true,
                    exp["near_wall_drag"])
                f_pore = pore_force_z_gap_pn(gap_true, cfg)

                # Optical restoring force: positive = away from membrane.
                f_opt = -k_actual * (gap_true - (focus_gap + drift))
                f_total = f_opt + f_pore

                # Euler-Maruyama overdamped Langevin update:
                # D = kBT/gamma; kBT ~= 4.114 pN*nm at 298 K.
                kbt = 4.114 * cfg["temperature_k"] / 298.0
                thermal_sigma = np.sqrt(max(0.0, 2.0 * kbt * dt_ms / gamma))
                dg = (f_total / gamma) * dt_ms
                if cfg["brownian"]:
                    dg += rng.normal(0.0, thermal_sigma)
                gap_true = max(0.0, gap_true + dg)

                measured_gap = gap_true + rng.normal(0.0, exp["detector_sigma_nm"])
                inferred_force = -k_nom * (measured_gap - focus_gap)

                all_rows.append({
                    "repeat": rep + 1,
                    "step": j + 1,
                    "t_ms": global_t_ms,
                    "command_gap_nm": float(target),
                    "focus_gap_nm": focus_gap,
                    "true_gap_nm": gap_true,
                    "measured_gap_nm": measured_gap,
                    "F_pore_true_pN": f_pore,
                    "F_opt_true_pN": f_opt,
                    "F_inferred_pN": inferred_force,
                    "k_actual_pN_per_nm": k_actual,
                })

    raw = pd.DataFrame(all_rows)

    # Analyze only the final half of each dwell to remove most settling transient.
    raw["hold_index"] = raw.groupby(["repeat", "step"]).cumcount()
    settle_cut = n_hold // 2
    settled = raw[raw["hold_index"] >= settle_cut].copy()

    per_repeat = (settled.groupby(["repeat", "step", "command_gap_nm"], as_index=False)
                  .agg(mean_true_gap_nm=("true_gap_nm", "mean"),
                       mean_measured_gap_nm=("measured_gap_nm", "mean"),
                       within_hold_sd_nm=("measured_gap_nm", "std"),
                       mean_inferred_force_pN=("F_inferred_pN", "mean"),
                       mean_true_pore_force_pN=("F_pore_true_pN", "mean")))

    summary = (per_repeat.groupby(["step", "command_gap_nm"], as_index=False)
               .agg(mean_actual_gap_nm=("mean_true_gap_nm", "mean"),
                    between_repeat_sd_nm=("mean_true_gap_nm", "std"),
                    mean_within_hold_sd_nm=("within_hold_sd_nm", "mean"),
                    mean_inferred_force_pN=("mean_inferred_force_pN", "mean"),
                    force_repeat_sd_pN=("mean_inferred_force_pN", "std"),
                    mean_true_pore_force_pN=("mean_true_pore_force_pN", "mean")))
    summary["bias_nm"] = summary["mean_actual_gap_nm"] - summary["command_gap_nm"]
    summary["rmse_like_nm"] = np.sqrt(summary["bias_nm"]**2 +
                                      summary["between_repeat_sd_nm"].fillna(0)**2 +
                                      summary["mean_within_hold_sd_nm"].fillna(0)**2)
    summary["step_resolvable_2sigma"] = (
        exp["step_nm"] > 2.0 * np.sqrt(
            summary["between_repeat_sd_nm"].fillna(0)**2 +
            summary["mean_within_hold_sd_nm"].fillna(0)**2)
    )
    return raw, summary


def simulate_sinusoidal_protocol(cfg: dict, exp: dict) -> tuple[pd.DataFrame, dict]:
    """Simulate sinusoidal trap-position or force excitation near the nanopore."""
    rng = np.random.default_rng(int(exp["sin_seed"]))
    f_hz = max(exp["sin_frequency_hz"], 1e-6)
    cycles = int(exp["sin_cycles"])
    samples_per_cycle = int(exp["samples_per_cycle"])
    n = max(200, cycles * samples_per_cycle)
    total_s = cycles / f_hz
    t_s = np.linspace(0.0, total_s, n, endpoint=False)
    dt_ms = (t_s[1] - t_s[0]) * 1000.0

    k = max(exp["sin_kz_pn_per_nm"], 1e-6)
    base_gap = exp["sin_base_gap_nm"]
    gamma = drag_pn_ms_per_nm(cfg["radius_nm"], cfg["viscosity_mpas"],
                              base_gap, exp["sin_near_wall_drag"])
    fc_hz = (k / gamma) * 1000.0 / (2.0 * np.pi)

    gap = float(base_gap)
    rows = []
    for i, ts in enumerate(t_s):
        phase = 2.0 * np.pi * f_hz * ts
        if exp["sin_drive_mode"] == "Move trap center":
            focus_gap = base_gap + exp["sin_position_amplitude_nm"] * np.sin(phase)
            external_drive = 0.0
        else:
            focus_gap = base_gap
            external_drive = exp["sin_force_amplitude_pn"] * np.sin(phase)

        # Optional commanded-focus noise and slow drift.
        focus_gap += rng.normal(0.0, exp["sin_command_sigma_nm"])
        focus_gap += exp["sin_drift_nm_per_min"] * (ts / 60.0)

        f_pore = pore_force_z_gap_pn(gap, cfg)
        f_opt = -k * (gap - focus_gap)
        f_total = f_opt + f_pore + external_drive

        kbt = 4.114 * cfg["temperature_k"] / 298.0
        thermal_sigma = np.sqrt(max(0.0, 2.0 * kbt * dt_ms / gamma))
        gap += (f_total / gamma) * dt_ms
        if cfg["brownian"]:
            gap += rng.normal(0.0, thermal_sigma)
        gap = max(0.0, gap)

        measured = gap + rng.normal(0.0, exp["sin_detector_sigma_nm"])
        rows.append({
            "t_s": ts,
            "drive_phase_rad": phase,
            "focus_gap_nm": focus_gap,
            "true_gap_nm": gap,
            "measured_gap_nm": measured,
            "F_pore_pN": f_pore,
            "F_opt_pN": f_opt,
            "F_external_drive_pN": external_drive,
            "F_total_pN": f_total,
        })

    df_sin = pd.DataFrame(rows)

    # Lock-in style single-frequency fit after discarding first 20%.
    fit = df_sin.iloc[int(0.2 * len(df_sin)):].copy()
    w = 2.0 * np.pi * f_hz
    X = np.column_stack([
        np.ones(len(fit)),
        np.sin(w * fit.t_s.to_numpy()),
        np.cos(w * fit.t_s.to_numpy())
    ])
    beta, *_ = np.linalg.lstsq(X, fit.measured_gap_nm.to_numpy(), rcond=None)
    amp = float(np.hypot(beta[1], beta[2]))
    # y = A sin(wt - phi) => sin coeff=A cos(phi), cos coeff=-A sin(phi)
    phase_lag_deg = float(np.degrees(np.arctan2(-beta[2], beta[1])))

    if exp["sin_drive_mode"] == "Move trap center":
        input_amp = max(exp["sin_position_amplitude_nm"], 1e-9)
        gain = amp / input_amp
    else:
        # Static displacement expected from F/k.
        input_amp = max(exp["sin_force_amplitude_pn"] / k, 1e-9)
        gain = amp / input_amp

    omega_per_ms = 2.0 * np.pi * f_hz / 1000.0
    phi_abs = abs(np.radians(phase_lag_deg))
    k_eff_phase = (gamma * omega_per_ms / max(np.tan(phi_abs), 1e-9)) if phi_abs > 1e-6 else np.inf
    if exp["sin_drive_mode"] == "Apply sinusoidal force":
        denom = max(amp, 1e-9)
        dynamic_mag = exp["sin_force_amplitude_pn"] / denom
        k_eff_amp = np.sqrt(max(0.0, dynamic_mag**2 - (gamma*omega_per_ms)**2))
    else:
        k_eff_amp = np.nan
    metrics = {
        "corner_frequency_hz": fc_hz,
        "response_amplitude_nm": amp,
        "phase_lag_deg": phase_lag_deg,
        "normalized_gain": gain,
        "gamma_pn_ms_per_nm": gamma,
        "k_eff_phase_pn_per_nm": k_eff_phase,
        "k_eff_amp_pn_per_nm": k_eff_amp,
        "k_interaction_phase_pn_per_nm": k_eff_phase-k if np.isfinite(k_eff_phase) else np.nan,
    }
    return df_sin, metrics


def step_precision_figure(raw: pd.DataFrame, summary: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=False, vertical_spacing=.18,
                        subplot_titles=("Commanded vs actual vertical gap",
                                        "Force inferred from bead displacement"))
    fig.add_trace(go.Scatter(
        x=summary.command_gap_nm, y=summary.mean_actual_gap_nm,
        mode="markers+lines", name="Actual mean gap",
        error_y=dict(type="data",
                     array=summary.between_repeat_sd_nm.fillna(0),
                     visible=True)), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=summary.command_gap_nm, y=summary.command_gap_nm,
        mode="lines", name="Ideal y=x", line=dict(dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=summary.command_gap_nm, y=summary.mean_inferred_force_pN,
        mode="markers+lines", name="Inferred force",
        error_y=dict(type="data",
                     array=summary.force_repeat_sd_pN.fillna(0),
                     visible=True)), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=summary.command_gap_nm, y=summary.mean_true_pore_force_pN,
        mode="lines", name="Underlying pore-force proxy", line=dict(dash="dot")),
        row=2, col=1)
    fig.update_layout(height=700, template="plotly_dark",
                      paper_bgcolor="#080d18", plot_bgcolor="#080d18",
                      font=dict(color="#ffffff"),
                      legend=dict(orientation="h", y=1.12, font=dict(color="#ffffff", size=12), bgcolor="rgba(8,13,24,.88)", bordercolor="#3b4966", borderwidth=1))
    fig.update_xaxes(title_text="Commanded particle-surface gap (nm)",
                     autorange="reversed", row=1, col=1)
    fig.update_xaxes(title_text="Commanded particle-surface gap (nm)",
                     autorange="reversed", row=2, col=1)
    fig.update_yaxes(title_text="Gap (nm)", row=1, col=1)
    fig.update_yaxes(title_text="Force (pN)", color="#ffffff", gridcolor="#3b4966", row=2, col=1)
    fig.update_xaxes(color="#ffffff", gridcolor="#3b4966")
    fig.update_yaxes(color="#ffffff", gridcolor="#3b4966")
    fig.update_annotations(font=dict(color="#ffffff", size=14))
    return fig


def sinusoidal_figure(df_sin: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.16,
                        subplot_titles=("Sinusoidal command and measured bead response",
                                        "Force components"))
    fig.add_trace(go.Scatter(x=df_sin.t_s, y=df_sin.focus_gap_nm,
                             name="Trap-center command"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_sin.t_s, y=df_sin.measured_gap_nm,
                             name="Measured particle gap"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_sin.t_s, y=df_sin.F_opt_pN,
                             name="Optical force"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_sin.t_s, y=df_sin.F_pore_pN,
                             name="Pore force"), row=2, col=1)
    if np.any(np.abs(df_sin.F_external_drive_pN.to_numpy()) > 0):
        fig.add_trace(go.Scatter(x=df_sin.t_s, y=df_sin.F_external_drive_pN,
                                 name="Applied sinusoidal force"), row=2, col=1)
    fig.update_layout(height=700, template="plotly_dark",
                      paper_bgcolor="#080d18", plot_bgcolor="#080d18",
                      font=dict(color="#ffffff"),
                      legend=dict(orientation="h", y=1.12, font=dict(color="#ffffff", size=12), bgcolor="rgba(8,13,24,.88)", bordercolor="#3b4966", borderwidth=1))
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Gap (nm)", row=1, col=1)
    fig.update_yaxes(title_text="Force (pN)", row=2, col=1)
    return fig



# ---------------------------------------------------------------------------
# Force-distance / velocity and single-particle occupancy analyses
# ---------------------------------------------------------------------------

def conservative_force_components_pn(gap_nm: float, cfg: dict, scales: dict) -> dict:
    """Phenomenological conservative-force decomposition for feasibility studies.

    The terms are deliberately separated to match the experimental logic. They
    are not quantitative predictions until replaced/calibrated with measured or
    COMSOL-derived force-distance relations.
    """
    d = max(float(gap_nm), 1.0)
    voltage_factor = cfg["voltage_mv"] / 120.0
    pressure_factor = cfg["pressure_mbar"] / 100.0
    f_dlvo = -scales["dlvo_amp_pn"] * np.exp(-d / max(scales["dlvo_decay_nm"], 1e-6))
    f_ekt = -scales["ekt_amp_pn"] * voltage_factor * np.exp(-d / max(scales["ekt_decay_nm"], 1e-6))
    f_perm = -scales["perm_amp_pn"] * pressure_factor * np.exp(-d / max(scales["perm_decay_nm"], 1e-6))
    f_shear = -scales["shear_amp_pn"] * pressure_factor * np.exp(-d / max(scales["shear_decay_nm"], 1e-6))
    return {"F_DLVO_pN": f_dlvo, "F_EKT_pN": f_ekt, "F_perm_pN": f_perm, "F_shear_pN": f_shear}


def simulate_velocity_force_scan(cfg: dict, exp: dict, scales: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate F_meas(d,v)=F_cons(d)+alpha(d)v with repeated noisy measurements."""
    rng = np.random.default_rng(int(exp["velocity_seed"]))
    gaps = np.arange(exp["velocity_start_gap_nm"], exp["velocity_end_gap_nm"] - 0.5*exp["velocity_gap_step_nm"], -exp["velocity_gap_step_nm"])
    velocities = np.array(exp["velocities_nm_s"], dtype=float)
    rows=[]
    for d in gaps:
        comps=conservative_force_components_pn(d,cfg,scales)
        f_cons=sum(comps.values())
        alpha = exp["alpha0_pn_s_per_nm"] * (1.0 + exp["alpha_wall_factor"] * cfg["radius_nm"] / max(d,2.0))
        for v in velocities:
            for r in range(int(exp["velocity_repeats"])):
                f_sq = alpha*v
                noise = rng.normal(0.0, exp["force_noise_pn"])
                f_meas = f_cons + f_sq + noise
                rows.append(dict(gap_nm=d, velocity_nm_s=v, repeat=r+1, F_meas_pN=f_meas,
                                 F_squeeze_pN=f_sq, F_cons_true_pN=f_cons, alpha_pN_s_per_nm=alpha, **comps))
    raw=pd.DataFrame(rows)
    fits=[]
    for d,g in raw.groupby("gap_nm"):
        x=g.velocity_nm_s.to_numpy(); y=g.F_meas_pN.to_numpy()
        slope,intercept=np.polyfit(x,y,1)
        pred=slope*x+intercept
        resid_sd=float(np.std(y-pred,ddof=1)) if len(y)>2 else 0.0
        fits.append(dict(gap_nm=d,F_cons_extrapolated_pN=intercept,alpha_fit_pN_s_per_nm=slope,
                         fit_residual_sd_pN=resid_sd,F_cons_true_pN=float(g.F_cons_true_pN.iloc[0])))
    return raw,pd.DataFrame(fits).sort_values("gap_nm",ascending=False)


def velocity_scan_figure(raw: pd.DataFrame, fits: pd.DataFrame) -> go.Figure:
    fig=make_subplots(rows=1,cols=2,subplot_titles=("Measured force vs approach velocity","Conservative force vs particle–surface gap"))
    for d,g in raw.groupby("gap_nm"):
        means=g.groupby("velocity_nm_s",as_index=False).F_meas_pN.mean()
        fig.add_trace(go.Scatter(x=means.velocity_nm_s,y=means.F_meas_pN,mode="markers+lines",name=f"gap {d:g} nm"),row=1,col=1)
    fig.add_trace(go.Scatter(x=fits.gap_nm,y=fits.F_cons_extrapolated_pN,mode="markers+lines",name="Extrapolated Fcons"),row=1,col=2)
    fig.add_trace(go.Scatter(x=fits.gap_nm,y=fits.F_cons_true_pN,mode="lines",line=dict(dash="dot"),name="Model Fcons"),row=1,col=2)
    fig.update_layout(height=520,template="plotly_dark",paper_bgcolor="#080d18",plot_bgcolor="#080d18",
                      font=dict(color="#ffffff"),legend=dict(orientation="h",y=1.18,font=dict(color="#ffffff",size=11),bgcolor="rgba(8,13,24,.9)"))
    fig.update_annotations(font=dict(color="#ffffff",size=14))
    fig.update_xaxes(color="#ffffff",gridcolor="#3b4966"); fig.update_yaxes(color="#ffffff",gridcolor="#3b4966")
    fig.update_xaxes(title_text="Approach velocity (nm/s)",row=1,col=1); fig.update_yaxes(title_text="Measured force (pN)",row=1,col=1)
    fig.update_xaxes(title_text="Gap (nm)",autorange="reversed",row=1,col=2); fig.update_yaxes(title_text="Force (pN)",row=1,col=2)
    return fig


def occupancy_probabilities(concentration_particles_per_ml: float, capture_volume_fl: float) -> dict:
    # 1 mL = 1e12 fL
    lam=max(0.0,concentration_particles_per_ml*capture_volume_fl/1e12)
    p0=np.exp(-lam); p1=lam*np.exp(-lam); p2=1.0-p0-p1
    return {"lambda":lam,"P0":p0,"P1":p1,"P2plus":max(0.0,p2)}


st.title("Nanopore × Optical Trap")
st.caption("Interactive overdamped-Langevin trajectory simulator · quick analytic model")

with st.sidebar:
    st.header("Simulation controls")
    st.subheader("1 · Beam geometry")
    axes = st.multiselect("Active trapping axes", ["x", "y", "z"], default=["z"],
                          help="Select one, two, or three directions.") or ["z"]
    powers, waists, focuses = {}, {}, {}
    for a in axes:
        with st.expander(f"{a.upper()}-axis beam", expanded=True):
            powers[a] = st.number_input(f"Power (mW) · {a}", 0.0, 500.0, 35.0, 1.0, key=f"power_{a}")
            waists[a] = st.number_input(f"Waist w₀ (µm) · {a}", 0.05, 10.0, 0.8, 0.05, key=f"waist_{a}")
            bc1, bc2, bc3 = st.columns(3)
            focuses[a] = (bc1.number_input("Focus x", -10.0, 10.0, 0.0, .1, key=f"fx_{a}"),
                          bc2.number_input("Focus y", -10.0, 10.0, 0.0, .1, key=f"fy_{a}"),
                          bc3.number_input("Focus z", -10.0, 10.0, 0.8, .1, key=f"fz_{a}"))
    wavelength = st.number_input("Wavelength (nm)", 350.0, 2000.0, 1064.0, 1.0)
    optical_scale = st.number_input("Optical-force calibration factor", 0.0, 100.0, 1.0, 0.1,
                                    help="Dimensionless multiplier for the analytic optical-force proxy.")
    st.subheader("2 · Particle & medium")
    radius = st.number_input("Particle radius (nm)", 5.0, 5000.0, 250.0, 10.0)
    deformable_particle = st.toggle("Deformable polystyrene particle", True,
                                    help="Qualitative volume-conserving deformation near the pore.")
    max_radial_strain_pct = st.number_input(
        "Maximum radial deformation (%)", 0.0, 40.0, 15.0, 1.0,
        disabled=not deformable_particle,
        help="Maximum allowed reduction of the particle radius in the x-y plane. "
             "This is a phenomenological setting, not a calibrated Young's-modulus calculation.")
    c1, c2 = st.columns(2)
    particle_n = c1.number_input("Particle n", 1.0, 4.5, 1.59, 0.01)
    medium_n = c2.number_input("Medium n", 1.0, 2.5, 1.333, 0.001)
    density = c1.number_input("Particle density (kg/m³)", 100.0, 20000.0, 1050.0, 10.0)
    viscosity = c2.number_input("Viscosity (mPa·s)", 0.01, 100.0, 0.89, 0.01)
    medium_density = st.number_input("Medium density (kg/m³)", 100.0, 5000.0, 997.0, 1.0,
                                     help="Used with particle density to calculate buoyancy-corrected gravity.")
    temperature = st.number_input("Temperature (K)", 200.0, 500.0, 298.0, 1.0)
    st.caption("Initial particle position relative to pore center (0, 0, 0)")
    pc1, pc2, pc3 = st.columns(3)
    initial_position = (pc1.number_input("Start x (µm)", -10.0, 10.0, 1.55, .1),
                        pc2.number_input("Start y (µm)", -10.0, 10.0, -.75, .1),
                        pc3.number_input("Start z (µm)", -2.0, 10.0, 2.2, .1))
    st.subheader("3 · Nanopore")
    pore_radius = st.number_input("Pore radius (nm)", 10.0, 5000.0, 400.0, 10.0)
    required_on_axis_pct = max(0.0, 100.0 * (1.0 - pore_radius / radius))
    if radius >= pore_radius:
        if deformable_particle and required_on_axis_pct <= max_radial_strain_pct:
            st.warning(f"Rigid particle is larger than the pore, but an on-axis radial deformation "
                       f"of {required_on_axis_pct:.1f}% can allow passage in this qualitative model.")
        else:
            st.info(f"Particle is larger than the pore and needs at least {required_on_axis_pct:.1f}% "
                    "on-axis radial deformation for translocation, above the selected limit. "
                    "Translocation is geometrically restricted, but the particle can still be used "
                    "as a non-translocating force probe above the pore.")
    else:
        st.caption(f"Radial passage clearance: {pore_radius - radius:.1f} nm")
    voltage = st.number_input("Voltage (mV)", -1000.0, 1000.0, 120.0, 5.0)
    pressure = st.number_input("Pressure (mbar)", -1000.0, 1000.0, 0.0, 1.0)
    hamaker = st.number_input("Hamaker A (×10⁻²¹ J)", 0.0, 100.0, 6.0, 0.5)
    pore_scale = st.number_input("Nanopore-force calibration factor", 0.0, 100.0, 1.0, 0.1,
                                 help="Dimensionless multiplier for the effective EP/DEP/EOF/pore-force proxy.")
    st.subheader("4 · Dynamics")
    brownian = st.toggle("Brownian motion", True)
    gravity_enabled = st.toggle("Gravity + buoyancy", True,
                                help="Uses effective weight (ρparticle − ρmedium)Vg.")
    gravity_direction = st.selectbox("Gravity direction", ["-z", "+z", "-x", "+x", "-y", "+y"],
                                     disabled=not gravity_enabled)
    duration = st.number_input("Duration (ms)", 1.0, 1000.0, 40.0, 5.0)
    dt = st.number_input("Requested time step (ms)", 0.001, 2.0, 0.05, 0.01, format="%.3f")
    seed = st.number_input("Random seed", 0, 999999, 42, 1)

    st.subheader("5 · Precision experiment assumptions")
    st.caption("These settings are intentionally separated so you can ask whether a 10 nm step is actually resolvable.")
    kz_pn_per_nm = st.number_input("Axial trap stiffness kz (pN/nm)",
                                   0.0001, 1.0, 0.03, 0.005, format="%.4f")
    command_sigma_nm = st.number_input("Trap/stage command jitter σ (nm)",
                                       0.0, 100.0, 2.0, 0.5)
    detector_sigma_nm = st.number_input("Position-detector noise σ (nm)",
                                        0.0, 100.0, 2.0, 0.5)
    repeatability_sigma_nm = st.number_input("Re-zero / run-to-run offset σ (nm)",
                                             0.0, 200.0, 5.0, 1.0)
    drift_nm_per_min = st.number_input("Slow drift scale (nm/min)",
                                       0.0, 500.0, 5.0, 1.0)
    stiffness_cv_pct = st.number_input("Trap-stiffness calibration CV (%)",
                                       0.0, 100.0, 5.0, 1.0)
    near_wall_drag = st.toggle("Include near-wall drag penalty", True)

    run = st.button("Run new trajectory", type="primary", use_container_width=True)

cfg = dict(radius_nm=radius, particle_n=particle_n, density=density, medium_density=medium_density,
           deformable_particle=deformable_particle, max_radial_strain_pct=max_radial_strain_pct,
           medium_n=medium_n, gravity_enabled=gravity_enabled, gravity_direction=gravity_direction,
           viscosity_mpas=viscosity, temperature_k=temperature, pore_radius_nm=pore_radius,
           voltage_mv=voltage, pressure_mbar=pressure, hamaker_1e21j=hamaker,
           optical_force_scale=optical_scale, pore_force_scale=pore_scale,
           duration_ms=duration, dt_ms=dt, seed=seed, brownian=brownian,
           wavelength_nm=wavelength, initial_position_um=initial_position,
           power_x_mw=powers.get("x", 0.0), power_y_mw=powers.get("y", 0.0),
           power_z_mw=powers.get("z", 0.0),
           waist_x_um=waists.get("x", .8), waist_y_um=waists.get("y", .8), waist_z_um=waists.get("z", .8),
           focus_x_um=focuses.get("x", (0., 0., .8)), focus_y_um=focuses.get("y", (0., 0., .8)),
           focus_z_um=focuses.get("z", (0., 0., .8)))

df = simulate(cfg, axes)
last = df.iloc[-1]; radial = math.hypot(last.x_um, last.y_um)
hover = analyze_hovering(df, radius/1000)
rigid_fit = radius < pore_radius
on_axis_deformable_fit = (deformable_particle and
                          required_on_axis_pct <= max_radial_strain_pct)
passage_possible = rigid_fit or on_axis_deformable_fit
current_clearance = pore_radius/1000 - last.particle_transverse_radius_um
captured = radial <= max(0.0, current_clearance) and 0 <= last.z_um < 1.1
translocated = passage_possible and last.z_um < 0
if not passage_possible:
    state = "Non-translocating force probe"
elif translocated:
    state = "Translocated"
elif hover["hovering"]:
    state = "Hovering near pore"
elif captured:
    state = "Captured above pore"
else:
    state = "Approaching"

m1, m2, m3, m4 = st.columns(4)
m1.metric("Particle state", state)
m2.metric("Final radial offset", f"{radial:.3f} µm")
m3.metric("Peak optical force", f"{df.F_opt_mag_pN.max():.3f} pN")
m4.metric("Peak radial deformation", f"{df.particle_radial_strain_pct.max():.1f}%")
if hover["hovering"]:
    st.success(f"Stable hovering detected at z = {hover['z_mean_um']:.3f} ± "
               f"{hover['z_std_um']:.3f} µm (mean ± Brownian fluctuation). "
               f"Mean net Fz = {hover['mean_fz_pn']:.3f} pN and "
               f"dFz/dz = {hover['restoring_slope_pn_per_um']:.3f} pN/µm.")
else:
    st.caption(f"Recent mean height: z = {hover['z_mean_um']:.3f} ± {hover['z_std_um']:.3f} µm; "
               f"mean net Fz = {hover['mean_fz_pn']:.3f} pN. No stable hovering state detected.")

st.subheader("Geometry layout")
g1, g2 = st.columns(2)
with g1: st.plotly_chart(geometry_figure(cfg, axes, "cross"), use_container_width=True)
with g2: st.plotly_chart(geometry_figure(cfg, axes, "top"), use_container_width=True)

st.subheader("Trajectory and force balance")
left, right = st.columns(2)
with left: st.plotly_chart(trajectory_figure(df, pore_radius/1000), use_container_width=True)
with right:
    st.plotly_chart(force_figure(df), use_container_width=True)
    st.caption("The relative optical/nanopore magnitude is set by the two calibration factors. "
               "It is not an experimental prediction until those factors are fitted to COMSOL or measured force data.")

with st.spinner("Rendering top-view and cross-section trajectories…"):
    top_gif_bytes = make_gif(df, pore_radius/1000, "top")
    cross_gif_bytes = make_gif(df, pore_radius/1000, "cross")
st.subheader("Animated trajectories")
top_anim, cross_anim = st.columns(2)
with top_anim:
    st.image(top_gif_bytes, caption="Top view (x–y) · nanopore centered at (0, 0, 0)",
             use_container_width=True)
with cross_anim:
    st.image(cross_gif_bytes, caption="Cross-section (x–z) · membrane and pore opening at z = 0",
             use_container_width=True)

info, actions = st.columns([1.25, .75])
with info:
    st.info(f"Initial particle: ({initial_position[0]:g}, {initial_position[1]:g}, {initial_position[2]:g}) µm\n\n"
            + "\n\n".join(f"{a.upper()} beam: {powers[a]:g} mW, w₀={waists[a]:g} µm, focus={focuses[a]}" for a in axes))
with actions:
    st.download_button("Download trajectory CSV", df.to_csv(index=False).encode(),
                       "nanopore_trajectory.csv", "text/csv", use_container_width=True)
    st.download_button("Download top-view GIF", top_gif_bytes, "nanopore_trajectory_top.gif",
                       "image/gif", use_container_width=True)
    st.download_button("Download cross-section GIF", cross_gif_bytes, "nanopore_trajectory_cross_section.gif",
                       "image/gif", use_container_width=True)


st.divider()
st.header("Optical-tweezer force measurement and precision analysis")
st.caption(
    "This section treats the optical trap as a calibrated spring and explicitly adds "
    "command jitter, detector noise, drift, stiffness uncertainty and Brownian motion. "
    "The distance below is the particle-surface gap: 100 nm means the bottom of the bead "
    "is 100 nm above the membrane plane."
)

tab_step, tab_velocity, tab_sin, tab_capture = st.tabs([
    "Step approach · Position & force resolution",
    "Velocity scan · Conservative force",
    "Sinusoidal response · Local stiffness",
    "Single-particle capture"
])

with tab_step:
    c1, c2, c3, c4 = st.columns(4)
    start_gap_nm = c1.number_input("Start gap (nm)", 1.0, 5000.0, 100.0, 10.0,
                                   key="step_start_gap")
    end_gap_nm = c2.number_input("End gap (nm)", 0.0, 5000.0, 20.0, 10.0,
                                 key="step_end_gap")
    step_nm = c3.number_input("Step size (nm)", 1.0, 500.0, 10.0, 1.0,
                              key="step_size")
    repeats = c4.number_input("Independent repeats", 2, 100, 10, 1,
                              key="step_repeats")

    d1, d2, d3 = st.columns(3)
    dwell_ms = d1.number_input("Hold time at each gap (ms)", 10.0, 10000.0, 300.0, 50.0,
                               key="step_dwell")
    precision_dt_ms = d2.number_input("Precision simulation dt (ms)", 0.001, 2.0, 0.05, 0.01,
                                      format="%.3f", key="step_dt")
    precision_seed = d3.number_input("Precision seed", 0, 999999, 1234, 1,
                                     key="step_seed")

    exp_step = dict(
        start_gap_nm=float(start_gap_nm),
        end_gap_nm=float(end_gap_nm),
        step_nm=float(step_nm),
        repeats=int(repeats),
        dwell_ms=float(dwell_ms),
        precision_dt_ms=float(precision_dt_ms),
        precision_seed=int(precision_seed),
        kz_pn_per_nm=float(kz_pn_per_nm),
        command_sigma_nm=float(command_sigma_nm),
        detector_sigma_nm=float(detector_sigma_nm),
        repeatability_sigma_nm=float(repeatability_sigma_nm),
        drift_nm_per_min=float(drift_nm_per_min),
        stiffness_cv_pct=float(stiffness_cv_pct),
        near_wall_drag=bool(near_wall_drag),
    )
    raw_step, step_summary = simulate_step_approach(cfg, exp_step)

    resolvable_fraction = float(step_summary.step_resolvable_2sigma.mean())
    worst_rmse = float(step_summary.rmse_like_nm.max())
    mean_bias = float(step_summary.bias_nm.abs().mean())
    force_cv = float(
        (step_summary.force_repeat_sd_pN.abs() /
         step_summary.mean_inferred_force_pN.abs().clip(lower=1e-9)).replace([np.inf], np.nan).median()
    )

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("10-nm-step resolvability",
              f"{100*resolvable_fraction:.0f}% of positions",
              help="A simple criterion: step size > 2σ of combined within-hold and repeat scatter.")
    s2.metric("Worst gap uncertainty", f"{worst_rmse:.1f} nm")
    s3.metric("Mean |gap bias|", f"{mean_bias:.1f} nm")
    s4.metric("Median force repeat CV",
              "—" if not np.isfinite(force_cv) else f"{100*force_cv:.1f}%")

    if resolvable_fraction >= 0.8 and worst_rmse < step_nm:
        st.success(
            "Under the selected assumptions, the commanded distance ladder is mostly distinguishable. "
            "The important quantity is not whether the trap can be commanded in 10 nm increments, "
            "but whether the actual bead-surface gap remains separated after Brownian motion, drift, "
            "external-force displacement and calibration uncertainty are included."
        )
    else:
        st.warning(
            "Under the selected assumptions, 10 nm spacing is not reliably resolved at all positions. "
            "Try increasing kz or dwell/repeats, or reducing drift / re-zero / detector error. "
            "This identifies a precision-limited regime for the selected measurement settings."
        )

    st.plotly_chart(step_precision_figure(raw_step, step_summary), use_container_width=True)
    st.dataframe(
        step_summary[[
            "command_gap_nm", "mean_actual_gap_nm", "bias_nm",
            "mean_within_hold_sd_nm", "between_repeat_sd_nm",
            "mean_inferred_force_pN", "force_repeat_sd_pN",
            "step_resolvable_2sigma"
        ]].round(3),
        use_container_width=True, hide_index=True
    )
    st.download_button(
        "Download step-approach summary CSV",
        step_summary.to_csv(index=False).encode(),
        "step_approach_precision_summary.csv", "text/csv"
    )


with tab_velocity:
    st.subheader("Force–distance measurement at multiple approach velocities")
    st.caption("At each gap, the model fits Fmeas(d,v) = Fcons(d) + α(d)v and extrapolates to v → 0 to estimate the conservative interaction force.")
    v1,v2,v3,v4=st.columns(4)
    velocity_start_gap_nm=v1.number_input("Start gap (nm)",20.0,5000.0,150.0,10.0,key="v_start")
    velocity_end_gap_nm=v2.number_input("End gap (nm)",1.0,5000.0,30.0,10.0,key="v_end")
    velocity_gap_step_nm=v3.number_input("Gap spacing (nm)",1.0,500.0,10.0,1.0,key="v_step")
    velocity_repeats=v4.number_input("Repeats per condition",2,100,8,1,key="v_rep")
    v5,v6,v7=st.columns(3)
    velocity_text=v5.text_input("Approach velocities (nm/s)","5, 10, 20, 50",key="v_list")
    force_noise_pn=v6.number_input("Force-readout noise σ (pN)",0.0,10.0,0.03,0.01,key="v_noise")
    velocity_seed=v7.number_input("Velocity-scan seed",0,999999,3456,1,key="v_seed")
    try:
        velocities_nm_s=[float(x.strip()) for x in velocity_text.split(",") if x.strip()]
    except ValueError:
        velocities_nm_s=[5.,10.,20.,50.]
        st.warning("Could not parse the velocity list; using 5, 10, 20, 50 nm/s.")
    st.markdown("**Phenomenological force components**")
    f1,f2,f3,f4=st.columns(4)
    dlvo_amp=f1.number_input("DLVO amplitude (pN)",0.0,20.0,0.30,0.05,key="dlvo_amp")
    ekt_amp=f2.number_input("Electrokinetic amplitude (pN)",0.0,20.0,0.25,0.05,key="ekt_amp")
    perm_amp=f3.number_input("Permeation amplitude (pN)",0.0,20.0,0.15,0.05,key="perm_amp")
    shear_amp=f4.number_input("Shear amplitude (pN)",0.0,20.0,0.10,0.05,key="shear_amp")
    g1,g2,g3,g4=st.columns(4)
    dlvo_decay=g1.number_input("DLVO decay length (nm)",1.0,1000.0,25.0,5.0,key="dlvo_decay")
    ekt_decay=g2.number_input("EKT decay length (nm)",1.0,1000.0,80.0,5.0,key="ekt_decay")
    perm_decay=g3.number_input("Permeation decay length (nm)",1.0,1000.0,100.0,5.0,key="perm_decay")
    shear_decay=g4.number_input("Shear decay length (nm)",1.0,1000.0,120.0,5.0,key="shear_decay")
    h1,h2=st.columns(2)
    alpha0=h1.number_input("Bulk squeeze coefficient α₀ (pN·s/nm)",0.0,1.0,0.001,0.0002,format="%.4f",key="alpha0")
    alpha_wall=h2.number_input("Near-wall squeeze enhancement",0.0,20.0,0.5,0.1,key="alpha_wall")
    vexp=dict(velocity_start_gap_nm=float(velocity_start_gap_nm),velocity_end_gap_nm=float(velocity_end_gap_nm),
              velocity_gap_step_nm=float(velocity_gap_step_nm),velocity_repeats=int(velocity_repeats),velocities_nm_s=velocities_nm_s,
              force_noise_pn=float(force_noise_pn),velocity_seed=int(velocity_seed),alpha0_pn_s_per_nm=float(alpha0),alpha_wall_factor=float(alpha_wall))
    scales=dict(dlvo_amp_pn=float(dlvo_amp),ekt_amp_pn=float(ekt_amp),perm_amp_pn=float(perm_amp),shear_amp_pn=float(shear_amp),
                dlvo_decay_nm=float(dlvo_decay),ekt_decay_nm=float(ekt_decay),perm_decay_nm=float(perm_decay),shear_decay_nm=float(shear_decay))
    raw_v,fit_v=simulate_velocity_force_scan(cfg,vexp,scales)
    st.plotly_chart(velocity_scan_figure(raw_v,fit_v),use_container_width=True)
    st.dataframe(fit_v.round(4),use_container_width=True,hide_index=True)
    st.download_button("Download velocity-scan summary CSV",fit_v.to_csv(index=False).encode(),"velocity_force_scan_summary.csv","text/csv")
    st.info("The separated DLVO/EKT/permeation/shear terms are sensitivity-analysis placeholders. Replace them with measured or COMSOL-derived force laws for quantitative interpretation.")

with tab_sin:
    a1, a2, a3, a4 = st.columns(4)
    sin_drive_mode = a1.selectbox("Drive mode",
                                  ["Move trap center", "Apply sinusoidal force"],
                                  key="sin_drive_mode")
    sin_base_gap_nm = a2.number_input("Mean gap (nm)", 1.0, 5000.0, 100.0, 10.0,
                                      key="sin_base_gap")
    sin_frequency_hz = a3.number_input("Frequency (Hz)", 0.1, 5000.0, 20.0, 5.0,
                                       key="sin_frequency")
    sin_cycles = a4.number_input("Cycles", 3, 100, 12, 1, key="sin_cycles")

    b1, b2, b3, b4 = st.columns(4)
    sin_position_amplitude_nm = b1.number_input(
        "Trap-center amplitude (nm)", 0.0, 1000.0, 20.0, 5.0,
        disabled=sin_drive_mode != "Move trap center", key="sin_pos_amp")
    sin_force_amplitude_pn = b2.number_input(
        "Force amplitude (pN)", 0.0, 100.0, 0.5, 0.1,
        disabled=sin_drive_mode != "Apply sinusoidal force", key="sin_force_amp")
    samples_per_cycle = b3.number_input("Samples / cycle", 20, 1000, 100, 10,
                                        key="sin_samples_cycle")
    sin_seed = b4.number_input("Sinusoidal seed", 0, 999999, 2222, 1,
                               key="sin_seed")

    exp_sin = dict(
        sin_drive_mode=sin_drive_mode,
        sin_base_gap_nm=float(sin_base_gap_nm),
        sin_frequency_hz=float(sin_frequency_hz),
        sin_cycles=int(sin_cycles),
        samples_per_cycle=int(samples_per_cycle),
        sin_kz_pn_per_nm=float(kz_pn_per_nm),
        sin_position_amplitude_nm=float(sin_position_amplitude_nm),
        sin_force_amplitude_pn=float(sin_force_amplitude_pn),
        sin_command_sigma_nm=float(command_sigma_nm),
        sin_detector_sigma_nm=float(detector_sigma_nm),
        sin_drift_nm_per_min=float(drift_nm_per_min),
        sin_near_wall_drag=bool(near_wall_drag),
        sin_seed=int(sin_seed),
    )
    df_sin, sin_metrics = simulate_sinusoidal_protocol(cfg, exp_sin)

    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Predicted corner frequency",
              f"{sin_metrics['corner_frequency_hz']:.1f} Hz")
    q2.metric("Measured response amplitude",
              f"{sin_metrics['response_amplitude_nm']:.2f} nm")
    q3.metric("Phase lag",
              f"{sin_metrics['phase_lag_deg']:.1f}°")
    q4.metric("Normalized amplitude gain",
              f"{sin_metrics['normalized_gain']:.3f}")

    r1, r2 = st.columns(2)
    r1.metric("Effective stiffness from phase",
              "—" if not np.isfinite(sin_metrics["k_eff_phase_pn_per_nm"]) else f"{sin_metrics['k_eff_phase_pn_per_nm']:.4f} pN/nm")
    r2.metric("Interaction stiffness estimate",
              "—" if not np.isfinite(sin_metrics["k_interaction_phase_pn_per_nm"]) else f"{sin_metrics['k_interaction_phase_pn_per_nm']:.4f} pN/nm",
              help="Estimated keff − calibrated optical-trap stiffness. Interpret quantitatively only after in-situ calibration.")

    ratio = sin_frequency_hz / max(sin_metrics["corner_frequency_hz"], 1e-12)
    if ratio < 0.3:
        st.success(
            "Drive frequency is well below the estimated corner frequency, so the bead should "
            "approximately follow the modulation quasi-statically under this model."
        )
    elif ratio < 1.0:
        st.warning(
            "Drive frequency is approaching the trap corner frequency. Amplitude attenuation "
            "and phase lag are becoming important; this can be useful for dynamic calibration, "
            "but it is no longer a quasi-static force scan."
        )
    else:
        st.error(
            "Drive frequency is at or above the estimated corner frequency. The particle cannot "
            "faithfully follow the imposed sinusoid; expect strong attenuation and phase delay."
        )

    st.plotly_chart(sinusoidal_figure(df_sin), use_container_width=True)
    st.download_button(
        "Download sinusoidal trajectory CSV",
        df_sin.to_csv(index=False).encode(),
        "sinusoidal_tweezer_response.csv", "text/csv"
    )

    st.caption(
        "For a real experiment, measure kz and detector conversion in situ. Near a surface, "
        "hydrodynamic drag and axial trap stiffness can change with height, so the displayed "
        "corner frequency is a sensitivity-analysis estimate rather than a final prediction."
    )


with tab_capture:
    st.subheader("Single-particle occupancy estimate")
    st.caption("A Poisson occupancy model estimates how often an effective capture volume contains zero, one, or multiple nanoparticles.")
    c1,c2=st.columns(2)
    concentration_particles_per_ml=c1.number_input("Particle number concentration (particles/mL)",1e3,1e15,1e8,1e7,format="%.3e",key="cap_conc")
    capture_volume_fl=c2.number_input("Effective capture volume (fL)",0.001,1e6,10.0,1.0,key="cap_vol")
    occ=occupancy_probabilities(float(concentration_particles_per_ml),float(capture_volume_fl))
    o1,o2,o3,o4=st.columns(4)
    o1.metric("Mean occupancy λ",f"{occ['lambda']:.4g}")
    o2.metric("P(0 particles)",f"{100*occ['P0']:.2f}%")
    o3.metric("P(exactly 1)",f"{100*occ['P1']:.2f}%")
    o4.metric("P(2 or more)",f"{100*occ['P2plus']:.2f}%")
    lam_grid=np.logspace(-3,1,250)
    occ_fig=go.Figure()
    occ_fig.add_trace(go.Scatter(x=lam_grid,y=lam_grid*np.exp(-lam_grid),name="P(exactly 1)"))
    occ_fig.add_trace(go.Scatter(x=lam_grid,y=1-np.exp(-lam_grid)*(1+lam_grid),name="P(2 or more)"))
    occ_fig.add_vline(x=occ['lambda'],line_dash="dash")
    occ_fig.update_layout(height=430,template="plotly_dark",paper_bgcolor="#080d18",plot_bgcolor="#080d18",font=dict(color="#ffffff"),
                          legend=dict(font=dict(color="#ffffff"),bgcolor="rgba(8,13,24,.9)"),xaxis_title="Mean occupancy λ",yaxis_title="Probability",xaxis_type="log")
    occ_fig.update_xaxes(color="#ffffff",gridcolor="#3b4966"); occ_fig.update_yaxes(color="#ffffff",gridcolor="#3b4966")
    st.plotly_chart(occ_fig,use_container_width=True)
    st.info("This is an occupancy estimate, not a complete optical-capture-rate model. Diffusion, flow, trap depth and residence time must be added for a quantitative capture probability.")


with st.expander("Model equation and limitations"):
    st.latex(r"\gamma\dot{\mathbf r}=\mathbf F_{opt}+\mathbf F_{EP}+\mathbf F_{DEP}+\mathbf F_{EOF}+\mathbf F_{VDW}+\mathbf F_h+\mathbf F_{g,eff}+\mathbf F_{Brownian}")
    st.markdown(r"**VDW** is the short-range van der Waals surface attraction represented by the Hamaker constant. "
                r"**Effective gravity** includes buoyancy: $\mathbf F_{g,eff}=(\rho_p-\rho_m)V\mathbf g$.")
    st.warning("Qualitative model only. Voltage and pressure are mapped to an effective localized pore force. "
               "For quantitative prediction, import COMSOL E, ∇E², velocity, and pressure fields and interpolate them along the trajectory.")
