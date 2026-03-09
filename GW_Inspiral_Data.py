# All the imports:
import numpy as np
import pandas as pd
from math import pi, cos
from scipy.integrate import solve_ivp
from sklearn.decomposition import PCA
from scipy.integrate import cumulative_trapezoid


## Goal of this project: 
"""
Build a classical ML surrogate that maps a small set of binary 
parameters (masses, initial separation, …) to the gravitational-wave 
signal or a reduced representation of that signal, using only 
physics-based synthetic data (quadrupole/inspiral approximations) 
and non-neural classical regression methods.
"""

# Part 1 is data generation:

# Defining the constants
c = 3e8				#[m/s^2]
G = 6.674e-11		#[m^3⋅kg^−1⋅s^−2]
M_sun = 1.989e30	#[kg]

# Let's calculate the orbital decay.
	# How the orbits of the two masses evolves with time and other factors
def Orbital_decay(r, m1_kg, m2_kg):

	# Calculate dr/dt
	dr_dt = (-64 * G**3 * m1_kg * m2_kg * (m1_kg+m2_kg))/(5 * c**5 * r**3)

	return dr_dt


def Orbital_evolution(r0, t_max, t_points, m1_kg, m2_kg):

	def da_dt_func(t, y):
		a = y[0]
		if a < 1e5:  # Stop if separation becomes very small (merger)
			return [0]
		return [Orbital_decay(a, m1_kg, m2_kg)]
	
	# Solve the ODE
	solution = solve_ivp(da_dt_func, 
			t_span=(0, t_max),
			y0=[r0],
			method='RK45',
			t_eval=t_points,  # Evaluate at exactly these times
			dense_output=False)
    
	t = solution.t
	r = solution.y[0]

	return r, t

def orbital_frequency(r, t, m1_kg, m2_kg):
	
	# Orbital Frequency: How fast the two masses going to orbit each other
	f_orb = (2*pi)**(-1) * np.sqrt((G*(m1_kg+m2_kg))/r**3)

	# f_gw = 2 * f_orb
	# omega_r  = pi * f_gw => 2 * pi * f_orb
	omega_r = 2 * pi * f_orb

	return f_orb, omega_r

# Now we need to calculate Amplitude of the Gravitational Waves aka strain (h)
	# The strain is a dimensionless quantity: the fractional change in length.
    # When a gravitational wave passes Earth, 
    # it stretches/compresses distances by a tiny fraction h.

def strain(r, t, m1_kg, m2_kg, distance_kpc):
	# I will be using the equations provided by:
	# https://www.tapir.caltech.edu/~teviet/Waves/gwave_details.html

	f_orb, omega_r = orbital_frequency(r, t, m1_kg, m2_kg)

	#chirp mass: Think of this as the parameter that controls 
		# How fast the binary spirals inward.
	chirp_mass = (m1_kg*m2_kg)**(3/5) / (m1_kg + m2_kg)**(1/5)

	# Easier to keep distance in kpc
	# We can convert it to m
	kpc_to_m = 3.086e22  # 1 kiloparsec in meters
	d = distance_kpc * kpc_to_m

	h = (4 * (G * chirp_mass)**(5/3) * (omega_r)**(2/3) ) / (c**4 * d)

	return h, chirp_mass


if __name__=="__main__":
	
	# Step 1 choose parameters:

	# m1, m2: Masses of the two objects (in solar masses, M☉)
	# r0: Initial orbital separation (in kilometers)
	# tc: Time to coalescence (when the merger happens)
	# distance_kpc: Distance to the observer

	n_samples = 1000
	m1 = np.random.uniform(1.2, 50, n_samples)				#[Solar Mass (kg)]
	m2 = np.random.uniform(1.2, 50, n_samples)				#[Solar Mass (kg)]
	r0 = np.random.uniform(5000e3, 100000e3, n_samples)		#[m]
	distance_kpc = np.random.uniform(10, 1e6, n_samples)	#[kilo Parsecs]

	# Converting the m1 and m2 to kg from solar masses
	m1_kg = m1 * M_sun
	m2_kg = m2 * M_sun

	def time_to_merger(r0, m1_kg, m2_kg):
		t_m = (5 * c**5 * r0**4) / (256 * G**3 * m1_kg * m2_kg * (m1_kg + m2_kg))
		return t_m

	data = []
	for i in range(n_samples):
		
		t_m = time_to_merger(r0[i], m1_kg[i], m2_kg[i])
		mu = (m1_kg[i]*m2_kg[i]) / (m1_kg[i] + m2_kg[i])

		# Create 20 evenly-spaced time points
		n_features = 17
		t_points = np.linspace(0.1*t_m, 0.99*t_m, n_features)

		r_evolved, t_evolved = Orbital_evolution(r0[i], t_m, t_points, m1_kg[i], m2_kg[i])
		
		h, chirp_mass = strain(r_evolved, t_evolved, m1_kg[i], m2_kg[i], distance_kpc[i])
		noise = np.random.normal(0, 0.1 * np.max(np.abs(h)), size=h.shape)
		h_noisy = h + noise
		
		f_orb, omega_r = orbital_frequency(r0[i], t_evolved, m1_kg[i], m2_kg[i])
		
		# Build row
		row = list(h_noisy) + [r0[i], t_m, chirp_mass, f_orb, distance_kpc[i], omega_r, mu]
		data.append(row)

	# Create proper column names
	columns = [f'h_noisy_t{j+1}' for j in range(17)] + ['r0 [m]', 't_merge [s]', 'chirp_mass [kg]', 'f_orb [Hz]', "distance_kpc [kpc]",
				"omega_r [rad/s]", "reduced_mass [kg]"]
	df = pd.DataFrame(data, columns=columns)

	print(f"The shape of the data: {df.shape}")  # Should be (1000, 23)

	df.to_csv('gravitational_wave_data.csv', index=False)
	print("The gravitational_wave_data.csv is saved")
