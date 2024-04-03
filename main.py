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
no_frames = 500

grid_size_x = 600
grid_size_y = 200
N_boundary = 5
size_x = grid_size_x + N_boundary * 2
size_y = grid_size_y + N_boundary * 2
dx = 2 / (size_x - 1)
dy = 2 / (size_y - 1)
dt = 5 * 1e-2

# TODO : Change viscosity to tensor
viscosity = 1.48 * 1e-5  # of air
# Typical diffiusion coefficient is in the range C(g*m**-3) hence between 300*1e-3 for
# fire source and 1-10*1e-3 far away from source (cooling down)
# 8. : Evaluation of Aerosol Fire Extinguishing Agent Using a Simple Diffusion Model : Chen-guang ZhuChun-xu LüJun Wang
# so it is tensor but for testing it is scalar
# TODO : Change diff to tensor
diff = 5 * 1e-4
avogardo = 6.022 * 10 * 1e23
propane_butan_molar_mass = 58.124 * 10 * 1e-3
propane_butan_molecular_mass = 102.22 * 10 * 1e3
mass = propane_butan_molecular_mass / avogardo
Q = 1  # chemical energy of detonation
# propane butane density is in range of 1.88 to 2.5 kg/m**3
deegres_of_freedom = 2

density = torch.zeros(size_x, size_y, device=device)
density_prev = torch.zeros(size_x, size_y, device=device)
u = torch.zeros(size_x, size_y, device=device)
v = torch.zeros(size_x, size_y, device=device)

# RANDOM WIND SPEEDS https://www.weather.gov/mfl/beaufort
r1 = -1.
r2 = 1.
u_prev = torch.zeros((size_x, size_y), device=device)
uniform_tensor = torch.zeros((grid_size_x - N_boundary, grid_size_y - N_boundary), device=device)
u_prev[N_boundary:grid_size_x, N_boundary:grid_size_y] = uniform_tensor.uniform_(r1, r2)

v_prev = torch.zeros((size_x, size_y), device=device)
v_prev[N_boundary:grid_size_x, N_boundary:grid_size_y] = uniform_tensor.uniform_(r1, r2)

pressure_prev = torch.zeros(size_x, size_y, device=device)
pressure = torch.zeros(size_x, size_y, device=device)

poisson_v_term = torch.zeros(size_x, size_y, device=device)


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


def velocity_to_temperature(velocity_matrix, mass, degrees_of_freedom):
    boltzmann_constant = 1.380649e-23
    kinetic_energy = 0.5 * mass * velocity_matrix ** 2
    Kelvin_matrix = (2 * kinetic_energy) / (degrees_of_freedom * boltzmann_constant)
    return Kelvin_matrix


def temperature_to_rgb(temperature):
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
# Dynamic density addition
def add_source_density(x, x0, dt, step):
    rg = (step + 2)
    if step + 1 > 15:
        rg = 15
    else:
        pass
    offset_vertical = int(grid_size_x / 3)
    center_x = grid_size_x // 2
    center_y = grid_size_y // 2
    idx_x_low = center_x - rg + offset_vertical
    idx_x_high = center_x + rg + offset_vertical
    idx_y_low = center_y - rg - 5
    idx_y_high = center_y + rg + 5
    idx_x = torch.randint(low=idx_x_low, high=idx_x_high, size=(1,))
    idx_y = torch.randint(low=idx_y_low, high=idx_y_high, size=(1,))

    x0[idx_x, idx_y] += 0.2 / rg
    x[idx_x, idx_y] += \
        dt * x0[idx_x, idx_y]

    x0[idx_x_low:idx_x_high, idx_y_low:idx_y_high] += 1. / rg

    x[idx_x_low:idx_x_high, idx_y_low:idx_y_high] += \
        dt * x0[idx_x_low:idx_x_high, idx_y_low:idx_y_high]
    return x, x0


# Static velocity field components
# TODO: Change to dynamic velocity field that accounts for gravity and temperature
# TODO: make velocity perturbations accorting to density
def add_source_u(density, x, x0, dt, step):
    rg = 3
    offset_vertical = int(grid_size_x / 3)
    center_x = grid_size_x // 2
    center_y = grid_size_y // 2
    idx_x_low = center_x - rg + offset_vertical
    idx_x_high = center_x + rg + offset_vertical
    idx_y_low = center_y - rg
    idx_y_high = center_y + rg

    # Fire
    x0 -= 5. * density

    # random air velocity
    x0[N_boundary:grid_size_x, N_boundary:grid_size_y] += uniform_tensor.uniform_(r1, r2)

    x[idx_x_low:idx_x_high, idx_y_low:idx_y_high] \
        = dt * x0[idx_x_low:idx_x_high, idx_y_low:idx_y_high]

    return x, x0


def add_source_v(density, x, x0, dt, step):
    rg = 10
    offset_vertical = int(grid_size_x / 3)
    center_x = grid_size_x // 2
    center_y = grid_size_y // 2
    idx_x_low = center_x - rg + offset_vertical
    idx_x_high = center_x + rg + offset_vertical
    idx_y_low = center_y - rg
    idx_y_high = center_y + rg

    # Fire
    x0 += 2 * (torch.rand(x0.shape, device=device) - 0.5) * density

    # random air velocity
    x0[N_boundary:grid_size_x, N_boundary:grid_size_y] += uniform_tensor.uniform_(r1, r2)
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
def transform_to_k_space(w2):
    w2_k = w2  # torch.fft.fft2(w2)
    return w2_k


# Step 3
# Implict method
#   w3(k) = w2(k)/(Identity_operator - v * dt * (i*k)**2)
def diffuse(z, x, x0, diff, dt):
    # imag = torch.tensor([-1], device=device)
    alpha = dt * diff * grid_size_x * grid_size_y
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
    h = 1. / max(grid_size_x, grid_size_y)
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


def vel_step(density, u, v, u_prev, v_prev, viscosity, dt, step):
    u, u_prev = add_source_u(density, u, u_prev, dt, step)
    v, v_prev = add_source_v(density, v, v_prev, dt, step)
    u, u_prev = SWAP(u, u_prev)
    u, u_prev = diffuse(1, u, u_prev, viscosity, dt)
    v, v_prev = SWAP(v, v_prev)
    v, v_prev = diffuse(2, v, v_prev, viscosity, dt)

    u, v, u_prev, v_prev = project(u, v, u_prev, v_prev)

    u, u_prev = SWAP(u, u_prev)
    v, v_prev = SWAP(v, v_prev)
    u, u_prev = advect(1, u, u_prev, u_prev, v_prev, dt)
    v, v_prev = advect(2, v, v_prev, u_prev, v_prev, dt)
    u, v, u_prev, v_prev = project(u, v, u_prev, v_prev)
    return u, v, u_prev, v_prev


def dens_step(density, density_prev, u, v, diff, dt, step):
    if step % 1 == 0:
        density, density_prev = add_source_density(density, density_prev, dt, step)
    else:
        pass
    # density, density_prev = SWAP(density, density_prev)
    density, density_prev = diffuse(0, density, density_prev, diff, dt)
    density, density_prev = SWAP(density, density_prev)
    density, density_prev = advect(0, density, density_prev, u, v, dt)
    return density, density_prev


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


def poisson_velocity_term(poisson_vel_term, density, dt, u, v, dx):
    a = 0
    b = a + 1
    c = b + 1
    d = -b
    e = -c
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


def update_grid(density, density_prev, u, u_prev, v, v_prev, pressure, poisson_v_term, dt, viscosity, diff, step):
    density, density_prev = dens_step(density, density_prev, u, v, diff, dt, step)
    u, v, u_prev, v_prev = vel_step(density, u, v, u_prev, v_prev, viscosity, dt, step)
    velocity_magnitude = torch.sqrt(u ** 2 + v ** 2)
    poisson_v_term = poisson_velocity_term(poisson_v_term, density, dt, u, v, dx)
    pressure = pressure_poisson(pressure, velocity_magnitude, 0.1)
    temperature = velocity_to_temperature(velocity_magnitude, mass, deegres_of_freedom)
    rgb = temperature_to_rgb(temperature)
    # rgb = torch.cat([rgb , density.unsqueeze(2)],dim=2)
    return density, density_prev, u, v, u_prev, v_prev, pressure, poisson_v_term, rgb


# Create animation
fig, [ax1, ax2, ax3, ax4, ax5] = plt.subplots(nrows=1, ncols=5)
ax1.set_axis_off()
ax2.set_axis_off()
ax3.set_axis_off()
ax4.set_axis_off()
ax5.set_axis_off()
ax1.set_title('Velocity field\n u component', size=8)
ax2.set_title('Velocity field\n v component', size=8)
ax3.set_title('Density', size=8)
ax4.set_title('Pressure Field', size=8)
ax5.set_title('RGB', size=8)
ims = []

for i in range(no_frames):
    # print(i)

    density, density_prev, u, v, u_prev, v_prev, pressure, poisson_v_term, rgb = \
        update_grid(density, density_prev, u,
                    u_prev, v,
                    v_prev, pressure,
                    poisson_v_term, dt,
                    viscosity, diff, i)

    u_component = ax1.imshow(u.cpu().numpy(), cmap='hot', animated=True)
    v_component = ax2.imshow(v.cpu().numpy(), cmap='hot', animated=True)
    d = ax3.imshow(density.cpu().numpy(), cmap='hot', animated=True)
    # d_norm = F.normalize(density.unsqueeze(2), p=1, dim=1)
    # , alpha = F.normalize(density, dim=0).cpu().numpy()
    pressure_field = ax4.imshow(pressure.cpu().numpy(), animated=True)
    rgb = ax5.imshow((rgb.cpu().numpy() * 255).astype(np.uint8))
    ims.append([d, u_component, v_component, pressure_field, rgb])

ani = animation.ArtistAnimation(fig, ims, interval=1, blit=True, repeat_delay=100)
# ani.save("pressure_field.gif", fps=25)
plt.show()

torch.cuda.empty_cache()
import sys

sys.modules[__name__].__dict__.clear()
import gc

gc.collect()
