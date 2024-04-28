# 1. : CFD Modeling of Fireimport matplotlib : Kevin B. McGrattan, Randall J. McDermott, Glenn P. Forney, Jason E. Floyd, Simo A. Hostikka, Howard R. Baum
# 2. : Physics-Based Combustion Simulation :MICHAEL B. NIELSEN, Autodesk, Denmark MORTEN BOJSEN-HANSEN, Autodesk, United Kingdom KONSTANTINOS STAMATELOS and ROBERT BRIDSON, Autodesk, Canada
# 3. : A mesh-free framework for high-order direct numerical simulations of combustion in  complex geometries
# 4. : Modelling and numerical simulation of combustion and multi-phase flows using finite volume methods on  unstructured meshes Jordi Muela Castro
# 5. : Stable Fluids : Jos Stam
# 6. : Fluid Control Using the Adjoint Method : Antoine McNamara Adrien Treuille Zoran Popovic Jos Stam
# 7. : Real-Time Fluid Dynamics for Games : Jos Stam
import time

import cv2
import matplotlib
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import animation
import torch.nn.functional as F

torch.cuda.synchronize()
CUDA_LAUNCH_BLOCKING = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
matplotlib.use('TkAgg')
plt.style.use('dark_background')
no_frames = 500
grid_size_x = 600
grid_size_y = 400
N_boundary = int(grid_size_x / 100)
size_x = grid_size_x + N_boundary * 2
size_y = grid_size_y + N_boundary * 2
dx = 2 / (size_x - 1)  # [m]
dy = 2 / (size_y - 1)
dt = 1 * 1e-2  # [s]
degrees_of_freedom = 2
# TODO : Change viscosity to tensor
viscosity = 1.48 * 1e-5  # of air
# Typical diffiusion coefficient is in the range C(g*m**-3) hence between 300*1e-3 for
# fire source and 1-10*1e-3 far away from source (cooling down)
# 8. : Evaluation of Aerosol Fire Extinguishing Agent Using a Simple Diffusion Model : Chen-guang ZhuChun-xu LüJun Wang
# so it is tensor but for testing it is scalar
# TODO : Change diff to tensor
diff = 1e-2
avogardo = 6.022 * 10 * 1e23
gas_constant = R = 8.314
boltzmann_constant = 1.380649 * 10e-23
gravity = 9.8

# Ratio of reaction for propan-butane is 2/1
propane_molecular_mass = 44.097 * 1e-3  # g/mol
butane_molecular_mass = 58.12 * 1e-3  # g/mol
oxygen_molecular_mass = 15.999 * 1e-3  # g/mol
co2_molecular_mass = 44.01 * 1e-3  # g/mol
h2o_molecular_mass = 18.01528 * 1e-3  # g/mol
no_of_oxygn_in_the_reaction_for_propane = 13 / 2
no_of_oxygn_in_the_reaction_for_butane = 5
no_of_carbon_dioxide_propane = 3
no_of_carbon_dioxide_butane = 4
no_of_h2o_propane = 4
no_of_h2o_butane = 5
no_of_co2 = no_of_carbon_dioxide_propane + no_of_carbon_dioxide_butane
no_oxygen = no_of_oxygn_in_the_reaction_for_propane + no_of_oxygn_in_the_reaction_for_butane
no_of_h2o = no_of_h2o_propane + no_of_h2o_butane
fuel_molecular_mass = propane_molecular_mass + butane_molecular_mass
oxidizer_molecular_mass = 2 * oxygen_molecular_mass * no_oxygen
product_molecular_mass = co2_molecular_mass * no_of_co2 + h2o_molecular_mass * no_of_h2o
PE_fuel_oxidizer_propane = 2219.9 * 1e3 / avogardo
PE_fuel_oxidizer_butane = 2657 * 1e3 / avogardo
PE_total = (PE_fuel_oxidizer_propane + PE_fuel_oxidizer_butane) / 2

Su_propane_butane_burning_velocity = 38.3 * 1e-2

grid_unit_volume = 1
# propane butane fuel_density is in range of 1.88 to 2.5 kg/m**3
deegres_of_freedom = 2

fuel_density = torch.zeros(size_x, size_y, device=device)
fuel_density_prev = torch.zeros(size_x, size_y, device=device)
oxidizer_density = torch.ones(size_x, size_y, device=device)
oxidizer_density_prev = torch.ones(size_x, size_y, device=device)
product_density = torch.zeros(size_x, size_y, device=device)
product_density_prev = torch.zeros(size_x, size_y, device=device)

u = torch.zeros(size_x, size_y, device=device)
v = torch.zeros(size_x, size_y, device=device)

# RANDOM WIND SPEEDS https://www.weather.gov/mfl/beaufort
r1 = -0.1
r2 = 0.1
u_prev = torch.zeros((size_x, size_y), device=device)
uniform_tensor = torch.zeros((grid_size_x - N_boundary, grid_size_y - N_boundary), device=device)
u_prev[N_boundary:grid_size_x, N_boundary:grid_size_y] = uniform_tensor.uniform_(r1, r2)

v_prev = torch.zeros((size_x, size_y), device=device)
v_prev[N_boundary:grid_size_x, N_boundary:grid_size_y] = uniform_tensor.uniform_(r1, r2)

pressure_prev = torch.full((size_x, size_y), 1., device=device)
pressure = torch.full((size_x, size_y), 1., device=device)

temperature_prev = torch.full((size_x, size_y), 293., device=device)
temperature = torch.full((size_x, size_y), 293., device=device)

poisson_v_term = torch.zeros(size_x, size_y, device=device)

mass_fuel = torch.full((size_x, size_y), fuel_molecular_mass, device=device)  # m3
mass_oxidizer = torch.full((size_x, size_y), oxidizer_molecular_mass, device=device)  # m3
mass_product = torch.full((size_x, size_y), product_molecular_mass, device=device)  # m3
mass_fuel_prev = torch.full((size_x, size_y), fuel_molecular_mass, device=device)  # m3  # m3
mass_oxidizer_prev = torch.full((size_x, size_y), oxidizer_molecular_mass, device=device)  # m3
mass_product_prev = torch.full((size_x, size_y), product_molecular_mass, device=device)  # m3


def set_bnd(Nx, Ny, b, x):
    if b == 1:
        x[0, 1:Ny + 1] = 0.  # -x[1, 1:Ny + 1]
        x[Nx + 1, 1:Ny + 1] = 0.  # -x[Nx, 1:Ny + 1]
    else:
        x[0, 1:Ny + 1] = 0.  # x[1, 1:Ny + 1]
        x[Nx + 1, 1:Ny + 1] = 0.  # x[Nx, 1:Ny + 1]

    if b == 2:
        x[1:Nx + 1, 0] = 0.  # -x[1:Nx + 1, 1]
        x[1:Nx + 1, Ny + 1] = 0.  # -x[1:Nx + 1, Ny]
    else:
        x[1:Nx + 1, 0] = 0.  # x[1:Nx + 1, 1]
        x[1:Nx + 1, Ny + 1] = 0.  # x[1:Nx + 1, Ny]

    x[0, 0] = 0.  # 0.5 * (x[1, 0] + x[0, 1])
    x[0, Ny + 1] = 0.  # 0.5 * (x[1, Ny + 1] + x[0, Ny])
    x[Nx + 1, 0] = 0.  # 0.5 * (x[Nx, 0] + x[Nx + 1, 1])
    x[Nx + 1, Ny + 1] = 0.  # 0.5 * (x[Nx, Ny + 1] + x[Nx + 1, Ny])
    return x


def SWAP(x, x0):
    x, x0 = x0, x
    return x, x0


def divisionByZero(numerator, denominator, eps=1e-8):
    numerator_mask = torch.abs(numerator) <= eps
    denominator_mask = torch.abs(denominator) <= eps
    zero_mask = numerator_mask & denominator_mask
    result = torch.zeros_like(numerator, device=device)
    result[~zero_mask] = numerator[~zero_mask] / denominator[~zero_mask]
    return result


def nan2zero(tensor, nan, pinf, ninf):
    tensor = torch.nan_to_num(tensor, nan=nan, posinf=pinf, neginf=ninf)
    return tensor


def ignite(temperature, step):
    rg = 25
    offset_vertical = int(grid_size_x / 3)
    center_x = grid_size_x // 2
    center_y = grid_size_y // 2
    idx_x_low = center_x - rg + offset_vertical
    idx_x_high = center_x + rg + offset_vertical
    idx_y_low = center_y - rg
    idx_y_high = center_y + rg
    ignite_temp = 273. + 1300.  # lighter temperature
    if step > 35:
        pass
    else:
        temperature[idx_x_low:idx_x_high, idx_y_low:idx_y_high] = ignite_temp

    return temperature


def combustion(fuel_density, oxidizer_density, product_density,
               u, v, pressure, temperature, temperature_prev,
               mass_fuel, mass_fuel_prev, mass_oxidizer, mass_oxidizer_prev,
               mass_product, mass_product_prev, deegres_of_freedom, step):
    temperature += ignite(temperature, step)
    d_low_fuel = 1e-2
    d_high_fuel = 1e3
    d_low_oxidizer = 1e-2
    d_high_oxidizer = 1e1
    chemical_potential_energy = 1.
    th_point = 273. + 400.  # KELVINS

    density_treshold_unburned_fuel = ((fuel_density >= d_low_fuel) & (fuel_density <= d_high_fuel))
    density_treshold_unburned_oxizdizer = ((oxidizer_density >= d_low_oxidizer) & (oxidizer_density <= d_high_oxidizer))
    above_temperature_treshold = (temperature >= th_point)
    conditions_met = above_temperature_treshold & density_treshold_unburned_fuel & density_treshold_unburned_oxizdizer
    u_burning = -35.  # Note : empirical maximum vertical velocity of burning | TODO : Normalization to physical real values needed
    v_burning = 35.  # Note : empirical maximum horizontal velocity of burning
    ratio_density = fuel_density / oxidizer_density
    ratio_density = nan2zero(ratio_density, 0, 0, 0)
    pwr_extraction = 1. / (1 + torch.exp(-1 * (ratio_density - 0.5)))
    horizontal_directivity = torch.rand(conditions_met.shape, device=device)
    horizontal_directivity = horizontal_directivity < 0.5
    horizontal_directivity = horizontal_directivity >= 0.5
    u[conditions_met] += u_burning * pwr_extraction[conditions_met] * dt
    v[conditions_met] += (horizontal_directivity[conditions_met].float() * v_burning - v_burning / 2) * dt
    product_density[conditions_met] += (fuel_density[conditions_met] + oxidizer_density[conditions_met]) * 0.5
    fuel_density[conditions_met] -= fuel_density[conditions_met]
    oxidizer_density[conditions_met] -= oxidizer_density[conditions_met]
    return u, v, fuel_density, oxidizer_density, product_density


def explosion():
    pass


def evaporation_cooling(oxidizer_density, u, v):
    vertical_directivity = torch.rand(oxidizer_density.shape, device=device)
    horizontal_directivity = torch.rand(oxidizer_density.shape, device=device)
    vertical_directivity = vertical_directivity < 0.5
    vertical_directivity = vertical_directivity >= 0.5
    horizontal_directivity = horizontal_directivity < 0.5
    horizontal_directivity = horizontal_directivity >= 0.5

    cooling_u_magnitude = 1.
    cooling_v_magnitude = 1.
    u[N_boundary:grid_size_x, N_boundary:grid_size_y] += oxidizer_density[N_boundary:grid_size_x,
                                                         N_boundary:grid_size_y] * (vertical_directivity.float()[
                                                                                    N_boundary:grid_size_x,
                                                                                    N_boundary:grid_size_y] * cooling_u_magnitude - cooling_u_magnitude / 2) * dt
    v[N_boundary:grid_size_x, N_boundary:grid_size_y] += oxidizer_density[N_boundary:grid_size_x,
                                                         N_boundary:grid_size_y] * (horizontal_directivity.float()[
                                                                                    N_boundary:grid_size_x,
                                                                                    N_boundary:grid_size_y] * cooling_v_magnitude - cooling_v_magnitude / 2) * dt
    return u, v, oxidizer_density


def radiative_cooling(fuel_density, oxidizer_density, product_density, u, v, temperature):
    d_low_product = 1e-3
    d_high_product = 1e1
    density_treshold_burned_product = ((product_density >= d_low_product) & (product_density <= d_high_product))
    th_point = 273. + 200.  # KELVINS
    above_temperature_treshold = (temperature >= th_point)
    conditions_met = density_treshold_burned_product & above_temperature_treshold
    u[conditions_met] += product_density[conditions_met] * (-u[conditions_met]) / 200
    v[conditions_met] += product_density[conditions_met] * (-v[conditions_met]) / 200
    return u, v, product_density


def radiative_heating(fuel_density, oxidizer_density, product_density, u, v, temperature):
    d_low_product = 1e1
    d_high_product = 20.
    density_treshold_burned_product = ((product_density >= d_low_product) & (product_density <= d_high_product))
    th_point = 273. + 400.  # KELVINS
    above_temperature_treshold = (temperature >= th_point)
    conditions_met = density_treshold_burned_product & above_temperature_treshold
    u[conditions_met] += product_density[conditions_met] * (u[conditions_met]) / 200
    v[conditions_met] += product_density[conditions_met] * (v[conditions_met]) / 200
    return u, v, product_density


def velocity2temperature(velocity_matrix, fuel_density, oxidizer_density, product_density, mass, degrees_of_freedom):
    kinetic_energy_of_one_particle = 0.5 * mass * velocity_matrix ** 2
    N_fuel_atoms = avogardo * fuel_density * grid_unit_volume / fuel_molecular_mass
    N_oxy_atoms = avogardo * oxidizer_density * grid_unit_volume / oxygen_molecular_mass
    N_product_atoms = avogardo * product_density * grid_unit_volume / product_molecular_mass
    N_atoms_per_unit_volume = N_fuel_atoms + N_oxy_atoms + N_product_atoms
    Kelvin_matrix = (degrees_of_freedom / 3 * boltzmann_constant) * (
            kinetic_energy_of_one_particle * N_atoms_per_unit_volume)
    # print(Kelvin_matrix.max().cpu())
    return Kelvin_matrix


def temperature2velocity(pressure, temperature, fuel_density, product_density, oxidizer_density,
                         mass_fuel, mass_oxidizer, mass_product,
                         degrees_of_freedom):
    kT = (boltzmann_constant * temperature)
    N_oxy_atoms = pressure * mass_oxidizer * grid_unit_volume / oxidizer_density / kT
    N_oxy_atoms = nan2zero(N_oxy_atoms, 0., 0., 0.)

    kinetic_energy_per_atom = (3 / degrees_of_freedom) * kT
    velocity_matrix_oxidizer = (2 * N_oxy_atoms * kinetic_energy_per_atom / mass_oxidizer) ** 0.5
    velocity_matrix_oxidizer = nan2zero(velocity_matrix_oxidizer, 0., 0., 0.)
    v = velocity_matrix_oxidizer - 2 * velocity_matrix_oxidizer * torch.rand(velocity_matrix_oxidizer.shape[0],
                                                                             velocity_matrix_oxidizer.shape[1],
                                                                             device=device)

    u = velocity_matrix_oxidizer - 2 * velocity_matrix_oxidizer * torch.rand(velocity_matrix_oxidizer.shape[0],
                                                                             velocity_matrix_oxidizer.shape[1],
                                                                             device=device)
    return u * dt, v * dt  # m/s


def temperature2rgb(temperature):
    temperature = temperature / 100
    red = torch.zeros_like(temperature)
    green = torch.zeros_like(temperature)
    blue = torch.zeros_like(temperature)

    if torch.is_tensor(temperature):
        temperature = temperature.float()

    if torch.any(temperature <= 66):
        red[temperature <= 66] = 255
        mask = temperature <= 66
        red[~mask] = temperature[~mask] - 60
        red[~mask] = 329.698727446 * (red[~mask] ** -0.1332047592)
        red = torch.clamp(red, 0, 255)

        green[mask] = temperature[mask]
        green[mask] = 99.4708025861 * torch.log(green[mask]) - 161.1195681661
        green[mask] = torch.clamp(green[mask], 0, 255)
    else:
        green = temperature - 60
        green = 288.1221695283 * (green ** -0.0755148492)
        green = torch.clamp(green, 0, 255)

    if torch.any(temperature >= 66):
        blue[temperature >= 66] = 255
    else:
        blue[temperature <= 19] = 0
        mask = (temperature > 19) & (temperature < 66)
        blue[mask] = temperature[mask] - 10
        blue[mask] = 138.5177312231 * torch.log(blue[mask]) - 305.0447927307
        blue[mask] = torch.clamp(blue[mask], 0, 255)

    return torch.stack((green, blue, red), dim=2)


# Step 1
# w1(x) = w0(x) + dt * f(x,t)
# Dynamic fuel_density addition
def add_fuel_density(x, x0, dt, step):
    rg = 25
    offset_vertical = int(grid_size_x / 3)
    center_x = grid_size_x // 2
    center_y = grid_size_y // 2
    idx_x_low = center_x - rg + offset_vertical
    idx_x_high = center_x + rg + offset_vertical
    idx_y_low = center_y - rg - 5
    idx_y_high = center_y + rg + 5
    idx_x = torch.randint(low=idx_x_low, high=idx_x_high, size=(1,))
    idx_y = torch.randint(low=idx_y_low, high=idx_y_high, size=(1,))

    x0[idx_x, idx_y] += (1.808 + 2.48) / 2
    x0[idx_x_low:idx_x_high, idx_y_low:idx_y_high] += (1.808 + 2.48) / 2  # Note :  propane + butane kg/m3
    x[idx_x_low:idx_x_high, idx_y_low:idx_y_high] += \
        dt * x0[idx_x_low:idx_x_high, idx_y_low:idx_y_high]
    return x, x0


def add_oxidiser_density(x, x0, dt, step):
    # air density 1.225 kg/m3
    xmean = x0[N_boundary:grid_size_x, N_boundary:grid_size_y].mean()
    # xx = torch.zeros_like(x0[N_boundary:grid_size_x, N_boundary:grid_size_y],device=device)
    if xmean > 1.225:  # Note : air dens 1.225 kg/m3
        xx = -0.1
    else:
        xx = 0.1
    x[N_boundary:grid_size_x, N_boundary:grid_size_y] += dt * xx
    return x, x0


# Static velocity field components
def add_source_u(fuel_density, x, x0, dt, step):
    rg = 25
    offset_vertical = int(grid_size_x / 3)
    center_x = grid_size_x // 2
    center_y = grid_size_y // 2
    idx_x_low = center_x - rg + offset_vertical
    idx_x_high = center_x + rg + offset_vertical
    idx_y_low = center_y - rg
    idx_y_high = center_y + rg

    fuel_speed = 15.
    # Fire
    x0 -= fuel_speed * fuel_density

    x[idx_x_low:idx_x_high, idx_y_low:idx_y_high] \
        = dt * x0[idx_x_low:idx_x_high, idx_y_low:idx_y_high]

    return x, x0


def add_source_v(fuel_density, x, x0, dt, step):
    rg = 2
    offset_vertical = int(grid_size_x / 3)
    center_x = grid_size_x // 2
    center_y = grid_size_y // 2
    idx_x_low = center_x - rg + offset_vertical
    idx_x_high = center_x + rg + offset_vertical
    idx_y_low = center_y - rg
    idx_y_high = center_y + rg

    x0 += 2 * (torch.rand(x0.shape, device=device) - 0.5) * fuel_density

    x[idx_x_low:idx_x_high, idx_y_low:idx_y_high] \
        = dt * x0[idx_x_low:idx_x_high, idx_y_low:idx_y_high]
    return x, x0


# Step 2
# w2(x) = w1(p(x-dt))
def advect(b, x, x0, u, v, dt):
    dt0 = dt * max(grid_size_x, grid_size_y)
    i, j = torch.meshgrid(torch.arange(1, grid_size_x, device=device), torch.arange(1, grid_size_y, device=device),
                          indexing='ij')
    X = i.float() - dt0 * u[i, j]
    Y = j.float() - dt0 * v[i, j]

    X = torch.clamp(X, 0.5, grid_size_x + 0.5)
    Y = torch.clamp(Y, 0.5, grid_size_y + 0.5)

    i0 = X.floor().long()
    i1 = i0 + 1
    j0 = Y.floor().long()
    j1 = j0 + 1

    s1 = X - i0.float()
    s0 = 1 - s1
    t1 = Y - j0.float()
    t0 = 1 - t1

    x[i, j] = s0 * (t0 * x0[i0, j0] + t1 * x0[i0, j1]) + s1 * (t0 * x0[i1, j0] + t1 * x0[i1, j1])
    x = set_bnd(grid_size_x, grid_size_y, b, x)
    return x, x0


# Step 2.5
# In fourier domain gradient operator laplasian is equivalent
# to multiplication by i*k where i = sqrt(-1)
def transform2k_space(w2):
    w2_k = w2  # torch.fft.fft2(w2)
    return w2_k


# Step 3
# Implict method
#   w3(k) = w2(k)/(Identity_operator - v * dt * (i*k)**2)
def diffuse(z, x, x0, diff, dt):
    # imag = torch.tensor([-1], device=device)
    alpha = dt * diff * grid_size_x * grid_size_y * diff
    a = 0
    b = a + 1
    c = b + 1
    d = -b
    e = -c
    for k in range(20):
        x[b:d, b:d] = (x0[b:d, b:d] + alpha *
                       (x[a:e, b:d] + x[c:, b:d] + x[b:d, a:e] +
                        x[b:d, c:])) / (1 + 4 * alpha)

    x = set_bnd(grid_size_x, grid_size_y, z, x)
    return x, x0


# Step 4
# w4(k) = w3(k) - (i*k) * q
# (i*k)**2 * q = (i*k) *w3(k)
def project(u, v, u_prev, v_prev):
    h = 1. / ((grid_size_x + grid_size_y) / 2)
    a = 0
    b = a + 1
    c = b + 1
    d = -b
    e = -c
    v_prev[b:d, b:d] = -0.5 * h * ((u[c:, b:d] - u[:e, b:d]) + (v[b:d, c:] - v[b:d, :e]))
    u_prev.fill_(0.)
    u_prev = set_bnd(grid_size_x, grid_size_y, 0, u_prev)
    v_prev = set_bnd(grid_size_x, grid_size_y, 0, v_prev)

    for k in range(20):
        u_prev[b:d, b:d] = (v_prev[b:d, b:d] + u_prev[:e, b:d] +
                            u_prev[c:, b:d] + u_prev[b:d, :e] +
                            u_prev[b:d, c:]) / 4.
        u_prev = set_bnd(grid_size_x, grid_size_y, 0, u_prev)

    u[b:d, b:d] = u[b:d, b:d] - 0.5 * (u_prev[c:, b:d] - u_prev[:e, b:d]) / h
    v[b:d, b:d] = v[b:d, b:d] - 0.5 * (u_prev[b:d, c:] - u_prev[b:d, :e]) / h
    u = set_bnd(grid_size_x, grid_size_y, 1, u)
    v = set_bnd(grid_size_x, grid_size_y, 2, v)
    return u, v, u_prev, v_prev


# Step 4.5
# back transform from k space to x space
def transform_to_x_space(w4_k):
    w4 = w4_k  # torch.fft.ifft2(w4_k)
    return w4


def vel_step(fuel_density, oxidizer_density, product_density, u, v, u_prev, v_prev, viscosity, dt, step):
    u, u_prev = add_source_u(fuel_density, u, u_prev, dt, step)
    v, v_prev = add_source_v(fuel_density, v, v_prev, dt, step)
    u, u_prev = SWAP(u, u_prev)
    v, v_prev = SWAP(v, v_prev)
    u, u_prev = diffuse(1, u, u_prev, viscosity, dt)
    v, v_prev = diffuse(2, v, v_prev, viscosity, dt)
    u, v, u_prev, v_prev = project(u, v, u_prev, v_prev)
    u, u_prev = SWAP(u, u_prev)
    v, v_prev = SWAP(v, v_prev)
    u, u_prev = advect(1, u, u_prev, u_prev, v_prev, dt)
    v, v_prev = advect(2, v, v_prev, u_prev, v_prev, dt)
    u, v, u_prev, v_prev = project(u, v, u_prev, v_prev)
    return u, v, u_prev, v_prev, fuel_density, oxidizer_density, product_density


def dens_step(fuel_density, fuel_density_prev, oxidizer_density, oxidizer_density_prev, product_density,
              product_density_prev, u, v,
              diff, dt, step):
    oxidizer_density, oxidizer_density_prev = add_oxidiser_density(oxidizer_density, oxidizer_density_prev, dt, step)
    fuel_density, fuel_density_prev = add_fuel_density(fuel_density, fuel_density_prev, dt, step)

    fuel_density, fuel_density_prev = SWAP(fuel_density, fuel_density_prev)
    oxidizer_density, oxidizer_density_prev = SWAP(oxidizer_density, oxidizer_density_prev)
    product_density, product_density_prev = SWAP(product_density, product_density_prev)

    fuel_density, fuel_density_prev = diffuse(0, fuel_density, fuel_density_prev, diff, dt)
    oxidizer_density, oxidizer_density_prev = diffuse(0, oxidizer_density, oxidizer_density_prev, diff, dt)
    product_density, product_density_prev = diffuse(0, product_density, product_density_prev, diff, dt)

    fuel_density, fuel_density_prev = SWAP(fuel_density, fuel_density_prev)
    oxidizer_density, oxidizer_density_prev = SWAP(oxidizer_density, oxidizer_density_prev)
    product_density, product_density_prev = SWAP(product_density, product_density_prev)

    fuel_density, fuel_density_prev = advect(0, fuel_density, fuel_density_prev, u, v, dt)
    oxidizer_density, oxidizer_density_prev = advect(0, oxidizer_density, oxidizer_density_prev, u, v, dt)
    product_density, product_density_prev = advect(0, product_density, product_density_prev, u, v, dt)

    return fuel_density, fuel_density_prev, oxidizer_density, oxidizer_density_prev, product_density, product_density_prev


def pressure_poisson(p, poisson_vel_term, l2_target):
    iter_diff = l2_target + 1
    n = 0
    a = 0
    b = a + 1
    c = b + 1
    d = -b
    e = -c
    while iter_diff > l2_target and n <= 500:
        pn = p.clone().detach()
        p[b:d, b:d] = (0.25 * (pn[b:d, c:] +
                               pn[b:d, :e] +
                               pn[c:, b:d] +
                               pn[:e, b:d]) -
                       poisson_vel_term[b:d, b:d])

        p = set_bnd(grid_size_x, grid_size_y, 0, p)

        if n % 10 == 0:
            iter_diff = torch.sqrt(torch.sum((p - pn) ** 2) / torch.sum(pn ** 2))

        n += 1

    return p


def poisson_velocity_term(poisson_vel_term, fuel_density, oxidizer_density, product_density, dt, u, v, dx):
    a = 0
    b = a + 1
    c = b + 1
    d = -b
    e = -c
    density = fuel_density + oxidizer_density + product_density
    poisson_vel_term[b:d, b:d] = (
            density[b:d, b:d] * dx / 16 *
            (2 / dt * (u[b:d, c:] -
                       u[b:d, :e] +
                       v[c:, b:d] -
                       v[:e, b:d]) -
             2 / dx * (u[c:, b:d] - u[:e, b:d]) *
             (v[b:d, c:] - v[b:d, :e]) -
             (u[b:d, c:] - u[b:d, :e]) ** 2 / dx -
             (v[c:, b:d] - v[:e, b:d]) ** 2 / dx)
    )

    return poisson_vel_term


def update_grid(fuel_density, fuel_density_prev, oxidizer_density,
                oxidizer_density_prev, product_density,
                product_density_prev, u, u_prev, v, v_prev,
                pressure, temperature, temperature_prev, mass_fuel, mass_oxidizer,
                mass_product, poisson_v_term,
                dt, viscosity, diff, step):
    # u_init, v_init = temperature2velocity(pressure, temperature, fuel_density, product_density, oxidizer_density,
    #                                       mass_fuel, mass_oxidizer, mass_product,
    #                                       degrees_of_freedom)
    # u += u_init
    # v += v_init
    temperature, temperature_prev = SWAP(temperature, temperature_prev)

    fuel_density, fuel_density_prev, oxidizer_density, oxidizer_density_prev, product_density, product_density_prev = \
        dens_step(fuel_density,
                  fuel_density_prev,
                  oxidizer_density,
                  oxidizer_density_prev,
                  product_density,
                  product_density_prev, u, v,
                  diff, dt, step)
    mass_fuel = fuel_density * grid_unit_volume
    mass_oxidizer = oxidizer_density * grid_unit_volume
    mass_product = product_density * grid_unit_volume
    u, v, u_prev, v_prev, fuel_density, oxidizer_density, product_density = vel_step(fuel_density, oxidizer_density,
                                                                                     product_density, u, v, u_prev,
                                                                                     v_prev, viscosity, dt, step)

    u, v, fuel_density, oxidizer_density, product_density = combustion(fuel_density, oxidizer_density, product_density,
                                                                       u, v, pressure, temperature, temperature_prev,
                                                                       mass_fuel, mass_fuel_prev, mass_oxidizer,
                                                                       mass_oxidizer_prev, mass_product,
                                                                       mass_product_prev,
                                                                       deegres_of_freedom, step)
    u, v, oxidizer_density = evaporation_cooling(oxidizer_density, u, v)
    u, v, product_density = radiative_cooling(fuel_density, oxidizer_density, product_density, u, v, temperature)
    u, v, product_density = radiative_heating(fuel_density, oxidizer_density, product_density, u, v, temperature)

    velocity_magnitude = torch.sqrt(u ** 2 + v ** 2)
    poisson_v_term = poisson_velocity_term(poisson_v_term, fuel_density, oxidizer_density, product_density, dt, u, v,
                                           dx)
    pressure = pressure_poisson(pressure, velocity_magnitude, 0.1)
    temperature = velocity2temperature(velocity_magnitude, fuel_density, oxidizer_density, product_density,
                                       mass_fuel + mass_oxidizer + mass_product, deegres_of_freedom)
    # u, u_prev = SWAP(u, u_prev)
    # v, v_prev = SWAP(v, v_prev)
    rgb = temperature2rgb(temperature)

    # rgb = torch.cat([rgb , fuel_density.unsqueeze(2)],dim=2)
    return fuel_density, fuel_density_prev, \
        oxidizer_density, oxidizer_density_prev, \
        product_density, product_density_prev, \
        u, v, u_prev, v_prev, pressure, \
        temperature, temperature_prev, \
        mass_fuel, mass_oxidizer, mass_product, \
        poisson_v_term, rgb


# Create animation
fig, [[ax1, ax2, ax3, ax4], [ax5, ax6, ax7, ax8]] = plt.subplots(nrows=2, ncols=4)
ax1.set_axis_off()
ax2.set_axis_off()
ax3.set_axis_off()
ax4.set_axis_off()
ax5.set_axis_off()
ax6.set_axis_off()
ax7.set_axis_off()
ax8.set_axis_off()
font_title_size = 6
ax1.set_title('Velocity field\n u component', size=font_title_size)
ax2.set_title('Velocity field\n v component', size=font_title_size)
ax3.set_title('Fuel\n Density', size=font_title_size)
ax4.set_title('Oxidizer\n Density', size=font_title_size)
ax5.set_title('Product\n Densitty', size=font_title_size)
ax6.set_title('Pressure\n Field', size=font_title_size)
ax7.set_title('Temperature \n Field (K)', size=font_title_size)
ax8.set_title('RGB', size=font_title_size)
ims = []

for i in range(no_frames):
    fuel_density, fuel_density_prev, \
        oxidizer_density, oxidizer_density_prev, \
        product_density, product_density_prev, \
        u, v, u_prev, v_prev, \
        pressure, temperature, temperature_prev, \
        mass_fuel, mass_oxidizer, mass__product, \
        poisson_v_term, rgb = \
        update_grid(fuel_density, fuel_density_prev,
                    oxidizer_density, oxidizer_density_prev,
                    product_density, product_density_prev, u,
                    u_prev, v,
                    v_prev, pressure, temperature, temperature_prev,
                    mass_fuel, mass_oxidizer, mass_product,
                    poisson_v_term, dt,
                    viscosity, diff, i)

    u_component = ax1.imshow(u.cpu().numpy(), animated=True)
    v_component = ax2.imshow(v.cpu().numpy(), cmap='terrain', animated=True)
    d = ax3.imshow(fuel_density.cpu().numpy(), cmap='hot', animated=True)
    ox2 = ax4.imshow(oxidizer_density.cpu().numpy(), cmap='cool', animated=True)
    # d_norm = F.normalize(fuel_density.unsqueeze(2), p=1, dim=1)
    # , alpha = F.normalize(fuel_density, dim=0).cpu().numpy()
    combustion_products = ax5.imshow(product_density.cpu().numpy(), cmap='rainbow', animated=True)
    pressure_field = ax6.imshow(pressure.cpu().numpy(), cmap='inferno', animated=True)
    temp = ax7.imshow((temperature.cpu().numpy()), cmap='plasma')
    rgb = ax8.imshow((rgb.cpu().numpy() * 255).astype(np.uint8))
    ims.append([d, ox2, combustion_products, u_component, v_component, pressure_field, temp, rgb])

ani = animation.ArtistAnimation(fig, ims, interval=1, blit=True, repeat_delay=100)
ani.save("fixed_sim.gif")
plt.show()

torch.cuda.empty_cache()
import sys

sys.modules[__name__].__dict__.clear()
import gc

gc.collect()
