# 1. : CFD Modeling of Fireimport matplotlib : Kevin B. McGrattan, Randall J. McDermott, Glenn P. Forney, Jason E. Floyd, Simo A. Hostikka, Howard R. Baum
# 2. : Physics-Based Combustion Simulation :MICHAEL B. NIELSEN, Autodesk, Denmark MORTEN BOJSEN-HANSEN, Autodesk, United Kingdom KONSTANTINOS STAMATELOS and ROBERT BRIDSON, Autodesk, Canada
# 3. : A mesh-free framework for high-order direct numerical simulations of combustion in  complex geometries
# 4. : Modelling and numerical simulation of combustion and multi-phase flows using finite volume methods on  unstructured meshes Jordi Muela Castro
# 5. : Stable Fluids : Jos Stam
# 6. : Fluid Control Using the Adjoint Method : Antoine McNamara Adrien Treuille Zoran Popovic Jos Stam
# 7. : Real-Time Fluid Dynamics for Games : Jos Stam
import matplotlib
import torch
import matplotlib.pyplot as plt
from matplotlib import animation

torch.cuda.synchronize()
CUDA_LAUNCH_BLOCKING = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
matplotlib.use('TkAgg')

grid_size = 40
boundary_length = 2
size = grid_size + 2
dx = dy = 0.1
dt = 0.004
viscosity = 0.1
buoyancy_factor = 1.

density = torch.zeros(size, size, device=device)
density_prev = torch.zeros(size, size, device=device)
u = torch.zeros(size, size, device=device)
v = torch.zeros(size, size, device=device)
u_prev = torch.zeros(size, size, device=device)
v_prev = torch.zeros(size, size, device=device)


def set_bnd(N, b, x):
    for i in range(1, N + 1):
        if b == 1:
            x[0, i] = -x[1, i]
            x[N + 1, i] = -x[N, i]
        else:
            x[0, i] = x[1, i]
            x[N + 1, i] = x[N, i]

        if b == 2:
            x[i, 0] = -x[i, 1]
            x[i, N + 1] = -x[i, N]
        else:
            x[i, 0] = x[i, 1]
            x[i, N + 1] = x[i, N]

    x[0, 0] = 0.5 * (x[1, 0] + x[0, 1])
    x[0, N + 1] = 0.5 * (x[1, N + 1] + x[0, N])
    x[N + 1, 0] = 0.5 * (x[N, 0] + x[N + 1, 1])
    x[N + 1, N + 1] = 0.5 * (x[N, N + 1] + x[N + 1, N])
    return x


def SWAP(x, x0):
    x, x0 = x0, x
    return x, x0


# Step 1
# w1(x) = w0(x) + dt * f(x,t)
def add_force(w0, dt):
    # low = grid_size // 2
    # high = grid_size // 2
    idx_x = torch.randint(low=grid_size // 2 - 5, high=grid_size // 2 + 5, size=(1,))
    idx_y = torch.randint(low=grid_size // 2 - 2, high=grid_size // 2 + 5, size=(1,))
    # w0[int(grid_size / 2), int(grid_size / 2)] += 1.
    w0[idx_x, idx_y] += 1.
    w1 = w0 + dt * w0[idx_x, idx_y]
    return w1


# Step 2
# w2(x) = w1(p(x-dt))
def advect(b, x, x0, u, v, dt):
    dt0 = dt * grid_size
    i, j = torch.meshgrid(torch.arange(1, grid_size, device=device), torch.arange(1, grid_size, device=device),
                          indexing='ij')

    X = i.float() - dt0 * u[i, j]
    Y = j.float() - dt0 * v[i, j]

    X = torch.clamp(X, 0.5, grid_size + 0.5)
    Y = torch.clamp(Y, 0.5, grid_size + 0.5)

    i0 = X.floor().long()
    i1 = i0 + 1
    j0 = Y.floor().long()
    j1 = j0 + 1

    s1 = X - i0.float()
    s0 = 1 - s1
    t1 = Y - j0.float()
    t0 = 1 - t1

    x[i, j] = s0 * (t0 * x0[i0, j0] + t1 * x0[i0, j1]) + s1 * (t0 * x0[i1, j0] + t1 * x0[i1, j1])
    x = set_bnd(grid_size, b, x)
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
def diffuse(b, x, x0, dt):
    # imag = torch.tensor([-1], device=device)
    alpha = dt * viscosity * grid_size * grid_size
    for k in range(20):
        x[1:-1, 1:-1] = (x0[1:-1, 1:-1] + alpha *
                         (x[0:-2, 1:-1] + x[2:, 1:-1] + x[1:-1, 0:-2] +
                          x[1:-1, 2:])) / (1 + 4 * alpha)
        x[0, :] = x[1, :]
        x[-1, :] = x[-2, :]
        x[:, 0] = x[:, 1]
        x[:, -1] = x[:, -2]

    x = set_bnd(grid_size, b, x)

    return x, x0


# Step 4
# w4(k) = w3(k) - (i*k) * q
# (i*k)**2 * q = (i*k) *w3(k)
def project(u, v, u_prev, v_prev):
    h = 1. / grid_size
    u_new = torch.zeros_like(u)
    v_new = torch.zeros_like(v)

    v_new[1:-1, 1:-1] = -0.5 * h * ((u[2:, 1:-1] - u[:-2, 1:-1]) + (v[1:-1, 2:] - v[1:-1, :-2]))

    u_prev.fill_(0)

    for k in range(20):
        u_prev[1:-1, 1:-1] = (v_prev[1:-1, 1:-1] + u_prev[:-2, 1:-1] + u_prev[2:, 1:-1] + u_prev[1:-1, :-2] + u_prev[
                                                                                                              1:-1,
                                                                                                              2:]) / 4.

    u_new[1:-1, 1:-1] = u[1:-1, 1:-1] - 0.5 * (u_prev[2:, 1:-1] - u_prev[:-2, 1:-1]) / h
    v_new[1:-1, 1:-1] = v[1:-1, 1:-1] - 0.5 * (u_prev[1:-1, 2:] - u_prev[1:-1, :-2]) / h

    return u_new, v_new, u_prev, v_prev


# Step 4.5
# back transform from k space to x space
def transform_to_x_space(w4_k):
    w4 = w4_k  # torch.fft.ifft2(w4_k)
    return w4


def vel_step(u, v, u_prev, v_prev, dt):
    u_prev = add_force(u_prev, dt)
    v_prev = add_force(v_prev, dt)

    u, u_prev = SWAP(u, u_prev)
    u, u_prev = diffuse(1, u, u_prev, dt)

    v, v_prev = SWAP(v, v_prev)
    v, v_prev = diffuse(2, v, v_prev, dt)
    # u, v, u_prev, v_prev = project(u, v, u_prev, v_prev)
    u, u_prev = SWAP(u, u_prev)
    v, v_prev = SWAP(v, v_prev)

    u, u_prev = advect(1, u, u_prev, u_prev, v_prev, dt)
    v, v_prev = advect(2, v, v_prev, u_prev, v_prev, dt)
    # u, v, u_prev, v_prev = project(u, v, u_prev, v_prev)

    return u, u_prev, v, v_prev


def dens_step(density, density_prev, u, v, dt):
    density_prev = add_force(density_prev, dt)
    density, density_prev = diffuse(0, density, density_prev, dt)
    density, density_prev = advect(0, density_prev, density, u, v, dt)
    return density_prev, density


def update_grid(density, density_prev, u, u_prev, v, v_prev, dt, viscosity):
    u, u_prev, v, v_prev = vel_step(u, v, u_prev, v_prev, dt)
    density, density_prev = dens_step(density, density_prev, u, v, dt)
    return density, density_prev, u, u_prev, v, v_prev


# Create animation
fig, ax = plt.subplots()
ax.set_axis_off()
ims = []

for i in range(1000):
    # print(i)
    density, density_prev, u, u_prev, v, v_prev = update_grid(density, density_prev, u, u_prev, v, v_prev, dt,
                                                              viscosity)
    im = ax.imshow(density.cpu().numpy(), cmap='hot', animated=True)
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=1, blit=True, repeat_delay=100)
ani.save("diffiusion_advection.gif", fps=100)
plt.show()

torch.cuda.empty_cache()
import sys

sys.modules[__name__].__dict__.clear()
import gc

gc.collect()
