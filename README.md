# Optical_Tweezer_Nanopore
# Nanopore–Optical Trap Particle Trajectory Simulator

An interactive Streamlit application for qualitatively simulating the motion of a micro/nanoparticle under the combined effects of optical trapping, nanopore-induced forces, van der Waals attraction, effective gravity, hydrodynamic drag, and Brownian motion.

The particle trajectory is calculated using an overdamped Langevin model and visualized through interactive 3D plots, force analysis, geometry views, and an animated GIF.

## Main Features

### Configurable Optical Trapping Beams

* Select one, two, or three optical beams.
* Choose the propagation direction of each beam:

  * \(x\)-axis
  * \(y\)-axis
  * \(z\)-axis
* Independently configure each beam:

  * Laser power
  * Beam waist
  * Focal position \((x_f,y_f,z_f)\)
* Set the laser wavelength.
* Adjust the optical-force calibration factor.

### Particle and Medium Parameters

Users can configure:

* Particle radius
* Particle refractive index
* Particle density
* Initial particle position
* Medium refractive index
* Medium density
* Dynamic viscosity
* Temperature

### Nanopore Parameters

The nanopore center is fixed at:

$$
(x,y,z)=(0,0,0)
$$

Available nanopore parameters include:

* Nanopore radius
* Applied voltage
* Pressure difference
* Hamaker constant
* Nanopore-force calibration factor

### Included Forces

The current model includes the following force contributions:

$$
\mathbf F_{\mathrm{total}}
=
\mathbf F_{\mathrm{opt}}
+\mathbf F_{\mathrm{pore}}
+\mathbf F_{\mathrm{VDW}}
+\mathbf F_{g,\mathrm{eff}}
$$

where:

* \(\mathbf F_{\mathrm{opt}}\): optical trapping force
* \(\mathbf F_{\mathrm{pore}}\): effective combined nanopore force representing electrophoresis, dielectrophoresis, electro-osmotic flow, and hydrodynamic effects
* \(\mathbf F_{\mathrm{VDW}}\): short-range van der Waals attraction
* \(\mathbf F_{g,\mathrm{eff}}\): gravity corrected for buoyancy

The effective gravitational force is calculated as:

$$
\mathbf F_{g,\mathrm{eff}}
=
(\rho_p-\rho_m)V\mathbf g
$$

Gravity can be enabled or disabled and assigned to any of the following directions:

$$
+x,\;-x,\;+y,\;-y,\;+z,\;-z
$$

### Brownian Motion

Brownian motion can be enabled to represent random thermal collisions between the particle and surrounding liquid molecules.

Users can control:

* Simulation duration
* Numerical time step
* Temperature
* Random seed

Using the same random seed and the same parameters produces a reproducible trajectory.

## Simulation Model

The particle motion is calculated using an overdamped Langevin equation:

$$
\gamma\frac{d\mathbf r}{dt}
=
\mathbf F_{\mathrm{total}}
+\mathbf F_{\mathrm{Brownian}}
$$

The Stokes drag coefficient is approximated by:

$$
\gamma=6\pi\eta r
$$

where:

* \(\eta\) is the dynamic viscosity of the medium
* \(r\) is the particle radius

The optical force is represented by an effective harmonic trapping model around each beam focus. Transverse confinement is stronger than axial confinement.

The nanopore force is currently represented by a spatially decaying analytical approximation centered at the nanopore.

## Visualizations

The application provides:

* Interactive 3D particle trajectory
* Particle position relative to the nanopore
* \(x-z\) cross-sectional geometry view
* \(x-y\) top-view geometry
* Optical beam positions, beam waists, powers, and focal points
* Initial particle position
* Signed net-force components:

  * \(F_x\)
  * \(F_y\)
  * \(F_z\)
* Force magnitudes separated by physical source:

  * Optical force
  * Nanopore force
  * Van der Waals force
  * Effective gravity
  * Total force
* Animated trajectory GIF displayed directly in the webpage

## Data Export

Simulation results can be downloaded as:

* Particle trajectory CSV
* Force data CSV
* Animated trajectory GIF

The CSV output includes particle position, total force, and individual force contributions at each simulation time step.


## Requirements

The `requirements.txt` file should contain:

```text
streamlit>=1.45,<2
numpy>=2.0,<3
pandas>=2.2,<4
plotly>=6.0,<7
matplotlib>=3.9,<4
pillow>=11,<13
```


## Important Model Limitations

This application is currently intended for qualitative exploration, visualization, educational demonstrations, and preliminary parameter studies.

The analytical force expressions and calibration factors have not been experimentally calibrated. Therefore, the calculated force magnitudes and trajectories should not be interpreted as quantitative predictions of a specific experimental system.

In particular, the relative magnitude of the optical and nanopore forces depends on user-defined calibration factors. A larger simulated nanopore force does not necessarily mean that the nanopore force will dominate in a real experiment.

For quantitative analysis, the analytical fields should be replaced with experimentally measured data or numerical field maps from software such as COMSOL Multiphysics. Relevant spatial field data may include:

$$
E_x,\;E_y,\;E_z,\;\nabla |E|^2,\;u_x,\;u_y,\;u_z,\;p
$$

Future versions may support direct COMSOL field-map upload and spatial interpolation.

## Planned Improvements

Possible future improvements include:

* COMSOL electric-field and flow-field import
* Separate electrophoretic, dielectrophoretic, electro-osmotic, and hydrodynamic force models
* More rigorous Rayleigh or Mie optical-force calculations
* Particle–membrane collision and adhesion models
* Multiple-particle simulations
* Monte Carlo trapping-probability calculations
* Potential-energy and equilibrium-position analysis
* 3D animated trajectory export
* Experimental parameter calibration

## Disclaimer

This software is a research and educational prototype. It should not be used as a substitute for experimentally validated calculations or full multiphysics simulations.

## Author

Xianzhe Zhang

## License

This project is provided for academic and research use. Add an appropriate open-source license, such as the MIT License, if you plan to distribute or reuse the code publicly.
