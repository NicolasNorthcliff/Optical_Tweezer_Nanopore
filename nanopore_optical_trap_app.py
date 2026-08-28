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
from PIL import Image
import streamlit as st

st.set_page_config(page_title="Nanopore × Optical Trap", page_icon="🔬", layout="wide")


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
        pos = pos + (f_det / gamma) * dt + brownian_step
        if pos[2] < -0.35:  # simple membrane collision boundary
            pos[2] = -0.35 + abs(pos[2] + 0.35) * 0.25

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
        })
    return pd.DataFrame(rows)


def trajectory_figure(df: pd.DataFrame, pore_radius_um: float) -> go.Figure:
    th = np.linspace(0, 2*np.pi, 90)
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=df.x_um, y=df.y_um, z=df.z_um, mode="lines",
                               name="Trajectory", line=dict(color="#45dfcb", width=5)))
    fig.add_trace(go.Scatter3d(x=[df.x_um.iloc[-1]], y=[df.y_um.iloc[-1]], z=[df.z_um.iloc[-1]],
                               mode="markers", name="Particle", marker=dict(size=7, color="#eafffb")))
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
    fig.add_scatter(x=[start[ia]], y=[start[ib]], mode="markers+text", name="Initial particle",
                    marker=dict(size=15, color="white", line=dict(color="#45dfcb", width=3)),
                    text=[f"start ({start[0]:g}, {start[1]:g}, {start[2]:g}) µm"], textposition="top right")
    fig.add_scatter(x=[0], y=[0], mode="markers+text", name="Origin",
                    marker=dict(size=7, symbol="x", color="#f4bd62"), text=["pore center (0,0,0)"],
                    textposition="bottom right")
    fig.update_layout(height=430, template="plotly_dark", margin=dict(l=10, r=10, t=45, b=15),
                      title="Top view (x–y)" if view == "top" else "Cross-section (x–z)",
                      xaxis=dict(title=labels[0], range=[-2.7, 2.7], zeroline=True, gridcolor="#26324c"),
                      yaxis=dict(title=labels[1], range=[-2.7, 2.7] if view == "top" else [-1.1, 3.2],
                                 scaleanchor="x", scaleratio=1, gridcolor="#26324c"),
                      paper_bgcolor="#080d18", plot_bgcolor="#080d18", showlegend=False)
    return fig


def force_figure(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.16,
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
    fig.update_layout(height=620, template="plotly_dark", margin=dict(l=35, r=15, t=55, b=30),
                      paper_bgcolor="#080d18", plot_bgcolor="#080d18",
                      font=dict(size=14, color="#ffffff"),
                      legend=dict(orientation="h", y=1.08, font=dict(color="#ffffff")))
    fig.update_annotations(font=dict(color="#ffffff", size=15))
    fig.update_xaxes(title_text="Time (ms)", gridcolor="#3b4966", color="#ffffff", row=2, col=1)
    fig.update_yaxes(title_text="Force (pN)", gridcolor="#3b4966", zerolinecolor="#ffffff", color="#ffffff", row=1, col=1)
    fig.update_yaxes(title_text="Magnitude (pN)", gridcolor="#3b4966", color="#ffffff", row=2, col=1)
    return fig


def make_gif(df: pd.DataFrame, pore_radius_um: float, view: str = "top") -> bytes:
    """Render either an x-y top view or an x-z cross-section trajectory GIF."""
    sample = np.linspace(1, len(df), 60, dtype=int)
    if view == "top":
        horizontal, vertical = "x_um", "y_um"
        x_label, y_label = "x (µm)", "y (µm)"
    else:
        horizontal, vertical = "x_um", "z_um"
        x_label, y_label = "x (µm)", "z (µm)"
    x_lim = max(2.2, np.abs(df[horizontal]).max() * 1.12)
    if view == "top":
        y_limits = (-x_lim, x_lim)
    else:
        z_min = min(-0.55, float(df.z_um.min()) - 0.25)
        z_max = max(2.2, float(df.z_um.max()) + 0.25)
        y_limits = (z_min, z_max)
    frames = []
    for end in sample:
        fig, ax = plt.subplots(figsize=(5, 5), dpi=90)
        fig.patch.set_facecolor("#080d18"); ax.set_facecolor("#080d18")
        ax.plot(df[horizontal].iloc[:end], df[vertical].iloc[:end], color="#45dfcb", lw=1.8)
        ax.scatter(df[horizontal].iloc[end-1], df[vertical].iloc[end-1],
                   s=70, c="#eafffb", edgecolors="#45dfcb", zorder=5)
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
    run = st.button("Run new trajectory", type="primary", use_container_width=True)

cfg = dict(radius_nm=radius, particle_n=particle_n, density=density, medium_density=medium_density,
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
captured = radial < pore_radius/1000*1.25 and last.z_um < 1.1

m1, m2, m3, m4 = st.columns(4)
m1.metric("Capture state", "Captured" if captured else "Approaching")
m2.metric("Final radial offset", f"{radial:.3f} µm")
m3.metric("Peak optical force", f"{df.F_opt_mag_pN.max():.3f} pN")
m4.metric("Drag coefficient", f"{6*np.pi*viscosity*radius/1000:.2f}")

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

with st.expander("Model equation and limitations"):
    st.latex(r"\gamma\dot{\mathbf r}=\mathbf F_{opt}+\mathbf F_{EP}+\mathbf F_{DEP}+\mathbf F_{EOF}+\mathbf F_{VDW}+\mathbf F_h+\mathbf F_{g,eff}+\mathbf F_{Brownian}")
    st.markdown(r"**VDW** is the short-range van der Waals surface attraction represented by the Hamaker constant. "
                r"**Effective gravity** includes buoyancy: $\mathbf F_{g,eff}=(\rho_p-\rho_m)V\mathbf g$.")
    st.warning("Qualitative model only. Voltage and pressure are mapped to an effective localized pore force. "
               "For quantitative prediction, import COMSOL E, ∇E², velocity, and pressure fields and interpolate them along the trajectory.")
