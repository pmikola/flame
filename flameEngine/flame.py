# 1. : CFD Modeling of Fireimport matplotlib : Kevin B. McGrattan, Randall J. McDermott, Glenn P. Forney, Jason E. Floyd, Simo A. Hostikka, Howard R. Baum
# 2. : Physics-Based Combustion Simulation :MICHAEL B. NIELSEN, Autodesk, Denmark MORTEN BOJSEN-HANSEN, Autodesk, United Kingdom KONSTANTINOS STAMATELOS and ROBERT BRIDSON, Autodesk, Canada
# 3. : A mesh-free framework for high-order direct numerical simulations of combustion in  complex geometries
# 4. : Modelling and numerical simulation of combustion and multi-phase flows using finite volume methods on  unstructured meshes Jordi Muela Castro
# 5. : Stable Fluids : Jos Stam
# 6. : Fluid Control Using the Adjoint Method : Antoine McNamara Adrien Treuille Zoran Popovic Jos Stam
# 7. : Real-Time Fluid Dynamics for Games : Jos Stam
import matplotlib
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import animation

class flame_sim(object):
    def __init__(self, no_frames=1000, grid_x=700, grid_y=400, dt=1 * 1e-2, diff=5e-3, viscosity=1.48 * 1e-5,
                 d_low_fuel_c=1e-2,
                 d_high_fuel_c=1e3,
                 d_low_oxidizer_c=1e-2,
                 d_high_oxidizer_c=1e1,
                 th_point_c=273. + 400.,
                 d_low_product_r_c=1e-3,
                 d_high_product_r_c=1e1,
                 th_point_r_c=273. + 200.,
                 d_low_product_r_h=1e1,
                 d_high_product_r_h=20.,
                 th_point_r_h=273. + 400.):

        torch.cuda.synchronize()
        matplotlib.use('TkAgg')
        plt.style.use('dark_background')
        # CUDA_LAUNCH_BLOCKING = 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.no_frames = no_frames
        self.grid_size_x = grid_x
        self.grid_size_y = grid_y
        self.N_boundary = int(self.grid_size_x / 100)
        self.size_x = self.grid_size_x + self.N_boundary * 2
        self.size_y = self.grid_size_y + self.N_boundary * 2
        self.dx = 2 / (self.size_x - 1)  # [m]
        self.dy = 2 / (self.size_y - 1)
        self.dt = dt  # [s]
        self.degrees_of_freedom = 2
        self.viscosity = viscosity  # of air
        self.diff = diff
        self.avogardo = 6.022 * 10 * 1e23
        self.gas_constant = R = 8.314
        self.boltzmann_constant = 1.380649 * 10e-23
        self.gravity = 9.8
        self.propane_molecular_mass = 44.097 * 1e-3  # g/mol
        self.butane_molecular_mass = 58.12 * 1e-3  # g/mol
        self.oxygen_molecular_mass = 15.999 * 1e-3  # g/mol
        self.co2_molecular_mass = 44.01 * 1e-3  # g/mol
        self.h2o_molecular_mass = 18.01528 * 1e-3  # g/mol
        self.no_of_oxygn_in_the_reaction_for_propane = 13 / 2
        self.no_of_oxygn_in_the_reaction_for_butane = 5
        self.no_of_carbon_dioxide_propane = 3
        self.no_of_carbon_dioxide_butane = 4
        self.no_of_h2o_propane = 4
        self.no_of_h2o_butane = 5
        self.no_of_co2 = self.no_of_carbon_dioxide_propane + self.no_of_carbon_dioxide_butane
        self.no_oxygen = self.no_of_oxygn_in_the_reaction_for_propane + self.no_of_oxygn_in_the_reaction_for_butane
        self.no_of_h2o = self.no_of_h2o_propane + self.no_of_h2o_butane
        self.fuel_molecular_mass = self.propane_molecular_mass + self.butane_molecular_mass
        self.oxidizer_molecular_mass = 2 * self.oxygen_molecular_mass * self.no_oxygen
        self.product_molecular_mass = self.co2_molecular_mass * self.no_of_co2 + self.h2o_molecular_mass * self.no_of_h2o
        self.PE_fuel_oxidizer_propane = 2219.9 * 1e3 / self.avogardo
        self.PE_fuel_oxidizer_butane = 2657 * 1e3 / self.avogardo
        self.PE_total = (self.PE_fuel_oxidizer_propane + self.PE_fuel_oxidizer_butane) / 2

        self.Su_propane_butane_burning_velocity = 38.3 * 1e-2
        self.grid_unit_volume = 1

        self.d_low_fuel = d_low_fuel_c
        self.d_high_fuel = d_high_fuel_c
        self.d_low_oxidizer = d_low_oxidizer_c
        self.d_high_oxidizer = d_high_oxidizer_c
        self.th_point_c = th_point_c  # KELVINS

        self.d_low_product_r_c = d_low_product_r_c
        self.d_high_product_r_c = d_high_product_r_c
        self.th_point_r_c = th_point_r_c  # KELVINS

        self.d_low_product_r_h = d_low_product_r_h
        self.d_high_product_r_h = d_high_product_r_h
        self.th_point_r_h = th_point_r_h  # KELVINS

        self.low_alpha = 3.
        self.high_alpha = 12.
        self.alpha_decay = 0.001

        self.fuel_density = torch.zeros(self.size_x, self.size_y, device=self.device)
        self.fuel_density_prev = torch.zeros(self.size_x, self.size_y, device=self.device)
        self.oxidizer_density = torch.ones(self.size_x, self.size_y, device=self.device)
        self.oxidizer_density_prev = torch.ones(self.size_x, self.size_y, device=self.device)
        self.product_density = torch.zeros(self.size_x, self.size_y, device=self.device)
        self.product_density_prev = torch.zeros(self.size_x, self.size_y, device=self.device)

        self.u = torch.zeros(self.size_x, self.size_y, device=self.device)
        self.v = torch.zeros(self.size_x, self.size_y, device=self.device)
        self.velocity_magnitude = torch.zeros(self.size_x, self.size_y, device=self.device)

        # RANDOM WIND SPEEDS https://www.weather.gov/mfl/beaufort
        self.r1 = -0.1
        self.r2 = 0.1
        self.u_prev = torch.zeros((self.size_x, self.size_y), device=self.device)
        self.uniform_tensor = torch.zeros((self.grid_size_x - self.N_boundary - self.N_boundary,
                                           self.grid_size_y - self.N_boundary - self.N_boundary),
                                          device=self.device)
        self.u_prev[self.N_boundary:self.grid_size_x - self.N_boundary,
        self.N_boundary:self.grid_size_y - self.N_boundary] = self.uniform_tensor.uniform_(self.r1,
                                                                                           self.r2)

        self.v_prev = torch.zeros((self.size_x, self.size_y), device=self.device)
        self.v_prev[self.N_boundary:self.grid_size_x - self.N_boundary,
        self.N_boundary:self.grid_size_y - self.N_boundary] = self.uniform_tensor.uniform_(self.r1,
                                                                                           self.r2)

        self.pressure_prev = torch.full((self.size_x, self.size_y), 1., device=self.device)
        self.pressure = torch.full((self.size_x, self.size_y), 1., device=self.device)

        self.temperature_prev = torch.full((self.size_x, self.size_y), 293., device=self.device)
        self.temperature = torch.full((self.size_x, self.size_y), 293., device=self.device)

        self.poisson_v_term = torch.zeros(self.size_x, self.size_y, device=self.device)

        self.mass_fuel = torch.full((self.size_x, self.size_y), self.fuel_molecular_mass, device=self.device)  # m3
        self.mass_oxidizer = torch.full((self.size_x, self.size_y), self.oxidizer_molecular_mass,
                                        device=self.device)  # m3
        self.mass_product = torch.full((self.size_x, self.size_y), self.product_molecular_mass,
                                       device=self.device)  # m3
        self.mass_fuel_prev = torch.full((self.size_x, self.size_y), self.fuel_molecular_mass,
                                         device=self.device)  # m3  # m3
        self.mass_oxidizer_prev = torch.full((self.size_x, self.size_y), self.oxidizer_molecular_mass,
                                             device=self.device)  # m3
        self.mass_product_prev = torch.full((self.size_x, self.size_y), self.product_molecular_mass,
                                            device=self.device)  # m3

        self.alpha = torch.zeros_like(self.temperature, device=self.device)
        self.rgb = torch.zeros_like(torch.stack((self.temperature, self.temperature, self.temperature), dim=2))

    @staticmethod
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

    @staticmethod
    def SWAP(x, x0):
        x, x0 = x0, x
        return x, x0

    @staticmethod
    def nan2zero(tensor, nan, pinf, ninf):
        tensor = torch.nan_to_num(tensor, nan=nan, posinf=pinf, neginf=ninf)
        return tensor

    def divisionByZero(self, numerator, denominator, eps=1e-8):
        numerator_mask = torch.abs(numerator) <= eps
        denominator_mask = torch.abs(denominator) <= eps
        zero_mask = numerator_mask & denominator_mask
        result = torch.zeros_like(numerator, device=self.device)
        result[~zero_mask] = numerator[~zero_mask] / denominator[~zero_mask]
        return result

    def ignite(self, temperature, step):
        rg = 25
        offset_vertical = int(self.grid_size_x / 3)
        center_x = self.grid_size_x // 2
        center_y = self.grid_size_y // 2
        idx_x_low = center_x - rg + offset_vertical
        idx_x_high = center_x + rg + offset_vertical
        idx_y_low = center_y - rg
        idx_y_high = center_y + rg
        ignite_temp = 273. + 10300.  # Note: lighter temperature (273. + 1300.)
        if step > 75:
            pass
        else:
            temperature[idx_x_low:idx_x_high, idx_y_low:idx_y_high] = ignite_temp
        return temperature

    def combustion(self, fuel_density, oxidizer_density, product_density,
                   u, v, temperature, step):
        temperature += self.ignite(temperature, step)
        density_treshold_unburned_fuel = ((fuel_density >= self.d_low_fuel) & (fuel_density <= self.d_high_fuel))
        density_treshold_unburned_oxizdizer = (
                (oxidizer_density >= self.d_low_oxidizer) & (oxidizer_density <= self.d_high_oxidizer))
        above_temperature_treshold = (temperature >= self.th_point_c)
        conditions_met = above_temperature_treshold & density_treshold_unburned_fuel & density_treshold_unburned_oxizdizer
        u_burning = -35.  # Note : empirical maximum vertical velocity of burning | TODO : Normalization to physical real values needed
        v_burning = 45.  # Note : empirical maximum horizontal velocity of burning
        ratio_density = fuel_density / oxidizer_density
        ratio_density = self.nan2zero(ratio_density, 0, 0, 0)
        pwr_extraction = 1. / (1 + torch.exp(-1 * (ratio_density - 0.5)))
        horizontal_directivity = torch.rand(conditions_met.shape, device=self.device)
        horizontal_directivity = horizontal_directivity < 0.5
        horizontal_directivity = horizontal_directivity >= 0.5
        u[conditions_met] += u_burning * pwr_extraction[conditions_met] * self.dt
        v[conditions_met] += (horizontal_directivity[conditions_met].float() * v_burning - v_burning / 2) * self.dt
        product_density[conditions_met] += (fuel_density[conditions_met] + oxidizer_density[conditions_met]) * 0.5
        fuel_density[conditions_met] -= fuel_density[conditions_met]
        oxidizer_density[conditions_met] -= oxidizer_density[conditions_met]
        return u, v, fuel_density, oxidizer_density, product_density

    def explosion(self):
        # TODO : Need to add exception handling for rapid explosion (expansion/compression) for enhancing stability of sim
        pass

    def evaporation_cooling(self, oxidizer_density, u, v):
        vertical_directivity = torch.rand(oxidizer_density.shape, device=self.device)
        horizontal_directivity = torch.rand(oxidizer_density.shape, device=self.device)
        vertical_directivity = vertical_directivity < 0.5
        vertical_directivity = vertical_directivity >= 0.5
        horizontal_directivity = horizontal_directivity < 0.5
        horizontal_directivity = horizontal_directivity >= 0.5
        cooling_u_magnitude = 0.2
        cooling_v_magnitude = 0.2
        u[self.N_boundary:self.grid_size_x - self.N_boundary,
        self.N_boundary:self.grid_size_y - self.N_boundary] += oxidizer_density[
                                                               self.N_boundary:self.grid_size_x - self.N_boundary,
                                                               self.N_boundary:self.grid_size_y - self.N_boundary] * (
                                                                       vertical_directivity.float()[
                                                                       self.N_boundary:self.grid_size_x - self.N_boundary,
                                                                       self.N_boundary:self.grid_size_y - self.N_boundary] * cooling_u_magnitude - cooling_u_magnitude / 2) * self.dt
        v[self.N_boundary:self.grid_size_x - self.N_boundary,
        self.N_boundary:self.grid_size_y - self.N_boundary] += oxidizer_density[
                                                               self.N_boundary:self.grid_size_x - self.N_boundary,
                                                               self.N_boundary:self.grid_size_y - self.N_boundary] * (
                                                                       horizontal_directivity.float()[
                                                                       self.N_boundary:self.grid_size_x - self.N_boundary,
                                                                       self.N_boundary:self.grid_size_y - self.N_boundary] * cooling_v_magnitude - cooling_v_magnitude / 2) * self.dt
        return u, v, oxidizer_density

    def radiative_cooling(self, fuel_density, oxidizer_density, product_density, u, v, temperature):

        density_treshold_burned_product = (
                (product_density >= self.d_low_product_r_c) & (product_density <= self.d_high_product_r_c))
        above_temperature_treshold = (temperature >= self.th_point_r_c)
        conditions_met = density_treshold_burned_product & above_temperature_treshold
        u[conditions_met] += product_density[conditions_met] * (-u[conditions_met]) / 100
        v[conditions_met] += product_density[conditions_met] * (-v[conditions_met]) / 150
        return u, v, product_density

    def radiative_heating(self, fuel_density, oxidizer_density, product_density, u, v, temperature):

        density_treshold_burned_product = (
                (product_density >= self.d_low_product_r_h) & (product_density <= self.d_high_product_r_h))
        above_temperature_treshold = (temperature >= self.th_point_r_h)
        conditions_met = density_treshold_burned_product & above_temperature_treshold
        u[conditions_met] += product_density[conditions_met] * (u[conditions_met]) / 250
        v[conditions_met] += product_density[conditions_met] * (v[conditions_met]) / 250
        return u, v, product_density

    def velocity2temperature(self, velocity_matrix, fuel_density, oxidizer_density, product_density, mass,
                             degrees_of_freedom):
        kinetic_energy_of_one_particle = 0.5 * mass * velocity_matrix ** 2
        N_fuel_atoms = self.avogardo * fuel_density * self.grid_unit_volume / self.fuel_molecular_mass
        N_oxy_atoms = self.avogardo * oxidizer_density * self.grid_unit_volume / self.oxygen_molecular_mass
        N_product_atoms = self.avogardo * product_density * self.grid_unit_volume / self.product_molecular_mass
        N_atoms_per_unit_volume = N_fuel_atoms + N_oxy_atoms + N_product_atoms
        Kelvin_matrix = (degrees_of_freedom / 3 * self.boltzmann_constant) * (
                kinetic_energy_of_one_particle * N_atoms_per_unit_volume)
        # print(Kelvin_matrix.max().cpu())
        return Kelvin_matrix

    def temperature2velocity(self, pressure, temperature, fuel_density, product_density, oxidizer_density,
                             mass_fuel, mass_oxidizer, mass_product,
                             degrees_of_freedom):
        kT = (self.boltzmann_constant * temperature)
        N_oxy_atoms = pressure * mass_oxidizer * self.grid_unit_volume / oxidizer_density / kT
        N_oxy_atoms = self.nan2zero(N_oxy_atoms, 0., 0., 0.)

        kinetic_energy_per_atom = (3 / degrees_of_freedom) * kT
        velocity_matrix_oxidizer = (2 * N_oxy_atoms * kinetic_energy_per_atom / mass_oxidizer) ** 0.5
        velocity_matrix_oxidizer = self.nan2zero(velocity_matrix_oxidizer, 0., 0., 0.)
        v = velocity_matrix_oxidizer - 2 * velocity_matrix_oxidizer * torch.rand(velocity_matrix_oxidizer.shape[0],
                                                                                 velocity_matrix_oxidizer.shape[1],
                                                                                 device=self.device)

        u = velocity_matrix_oxidizer - 2 * velocity_matrix_oxidizer * torch.rand(velocity_matrix_oxidizer.shape[0],
                                                                                 velocity_matrix_oxidizer.shape[1],
                                                                                 device=self.device)
        return u * self.dt, v * self.dt  # m/s

    def temperature2rgb(self, temperature):
        temperature = temperature / 100
        red = torch.zeros_like(temperature)
        green = torch.zeros_like(temperature)
        blue = torch.zeros_like(temperature)
        alpha_mask = (temperature >= self.low_alpha) & (temperature <= self.high_alpha)
        self.alpha[alpha_mask] = torch.exp(-(temperature[alpha_mask] - self.low_alpha) / self.alpha_decay)
        self.alpha[temperature < self.low_alpha] = 0
        self.alpha[temperature > self.high_alpha] = 1
        self.alpha = torch.clamp(self.alpha, 0, 1)

        if torch.is_tensor(temperature):
            temperature = temperature.float()

        if torch.any(temperature <= 66):
            red[temperature <= 66] = 255
            mask = temperature <= 66
            red[~mask] = temperature[~mask] - 60
            red[~mask] = 329.698727446 * (red[~mask] ** -0.1332047592)

            green[mask] = temperature[mask]
            green[mask] = 99.4708025861 * torch.log(green[mask]) - 161.1195681661
        else:
            green = temperature - 60
            green = 288.1221695283 * (green ** -0.0755148492)

        if torch.any(temperature >= 66):
            blue[temperature >= 66] = 255
        else:
            blue[temperature <= 19] = 0
            mask = (temperature > 19) & (temperature < 66)
            blue[mask] = temperature[mask] - 10
            blue[mask] = 138.5177312231 * torch.log(blue[mask]) - 305.0447927307

        red[temperature < self.low_alpha] = 0
        blue[temperature < self.low_alpha] = 0
        green[temperature < self.low_alpha] = 0

        return torch.clamp(torch.stack((red, green, blue), dim=2), 0, 255), self.alpha

    # w1(x) = w0(x) + dt * f(x,t)
    # Dynamic fuel_density addition
    def add_fuel_density(self, x, x0, dt, step):
        rg = 25
        offset_vertical = int(self.grid_size_x / 3)
        center_x = self.grid_size_x // 2
        center_y = self.grid_size_y // 2
        idx_x_low = center_x - rg + offset_vertical
        idx_x_high = center_x + rg + offset_vertical
        idx_y_low = center_y - rg - 5
        idx_y_high = center_y + rg + 5
        idx_x = torch.randint(low=idx_x_low, high=idx_x_high, size=(1,))
        idx_y = torch.randint(low=idx_y_low, high=idx_y_high, size=(1,))

        x0[idx_x, idx_y] += (1.808 + 2.48)
        x0[idx_x_low:idx_x_high, idx_y_low:idx_y_high] += (1.808 + 2.48)  # Note :  propane + butane kg/m3
        x[idx_x_low:idx_x_high, idx_y_low:idx_y_high] += \
            dt * x0[idx_x_low:idx_x_high, idx_y_low:idx_y_high]
        return x, x0

    def add_oxidiser_density(self, x, x0, dt, step):
        # air density 1.225 kg/m3
        xmean = x0[self.N_boundary:self.grid_size_x - self.N_boundary,
                self.N_boundary:self.grid_size_y - self.N_boundary].mean()
        # xx = torch.zeros_like(x0[N_boundary:grid_size_x, N_boundary:grid_size_y],device=device)
        if xmean > 1.225:  # Note : air dens 1.225 kg/m3
            xx = -0.1225
        else:
            xx = 0.1225
        x[self.N_boundary:self.grid_size_x - self.N_boundary,
        self.N_boundary:self.grid_size_y - self.N_boundary] += dt * xx
        if x[self.N_boundary:self.grid_size_x - self.N_boundary,
           self.N_boundary:self.grid_size_y - self.N_boundary].mean() < 1:
            x[self.N_boundary:self.grid_size_x - self.N_boundary,
            self.N_boundary:self.grid_size_y - self.N_boundary] += dt * 1.225
        else:
            pass
        return x, x0

    # Static velocity field components
    def add_source_u(self, fuel_density, x, x0, dt, step):
        rg = 25
        offset_vertical = int(self.grid_size_x / 3)
        center_x = self.grid_size_x // 2
        center_y = self.grid_size_y // 2
        idx_x_low = center_x - rg + offset_vertical
        idx_x_high = center_x + rg + offset_vertical
        idx_y_low = center_y - rg
        idx_y_high = center_y + rg

        fuel_speed = 5.
        # Fire
        x0 -= fuel_speed * fuel_density

        x[idx_x_low:idx_x_high, idx_y_low:idx_y_high] \
            = dt * x0[idx_x_low:idx_x_high, idx_y_low:idx_y_high]

        x[self.N_boundary:self.grid_size_x - self.N_boundary,
        self.N_boundary:self.grid_size_y - self.N_boundary] += dt * self.gravity / 2000

        return x, x0

    def add_source_v(self, fuel_density, x, x0, dt, step):
        rg = 2
        offset_vertical = int(self.grid_size_x / 3)
        center_x = self.grid_size_x // 2
        center_y = self.grid_size_y // 2
        idx_x_low = center_x - rg + offset_vertical
        idx_x_high = center_x + rg + offset_vertical
        idx_y_low = center_y - rg
        idx_y_high = center_y + rg

        x0 += 2 * (torch.rand(x0.shape, device=self.device) - 0.5) * fuel_density

        x[idx_x_low:idx_x_high, idx_y_low:idx_y_high] \
            = dt * x0[idx_x_low:idx_x_high, idx_y_low:idx_y_high]
        return x, x0

    # Step 2
    # w2(x) = w1(p(x-dt))
    def advect(self, b, x, x0, u, v, dt):
        dt0 = dt * max(self.grid_size_x, self.grid_size_y)
        i, j = torch.meshgrid(torch.arange(1, self.grid_size_x, device=self.device),
                              torch.arange(1, self.grid_size_y, device=self.device),
                              indexing='ij')
        X = i.float() - dt0 * u[i, j]
        Y = j.float() - dt0 * v[i, j]

        X = torch.clamp(X, 0.5, self.grid_size_x + 0.5)
        Y = torch.clamp(Y, 0.5, self.grid_size_y + 0.5)

        i0 = X.floor().long()
        i1 = i0 + 1
        j0 = Y.floor().long()
        j1 = j0 + 1

        s1 = X - i0.float()
        s0 = 1 - s1
        t1 = Y - j0.float()
        t0 = 1 - t1

        x[i, j] = s0 * (t0 * x0[i0, j0] + t1 * x0[i0, j1]) + s1 * (t0 * x0[i1, j0] + t1 * x0[i1, j1])
        x = self.set_bnd(self.grid_size_x, self.grid_size_y, b, x)
        return x, x0

    # Step 2.5
    # In fourier domain gradient operator laplasian is equivalent
    # to multiplication by i*k where i = sqrt(-1)
    def transform2k_space(self, w2):
        w2_k = w2  # torch.fft.fft2(w2)
        # TODO : write this method for comparision
        return w2_k

    # Step 3
    # Implict method
    #   w3(k) = w2(k)/(Identity_operator - v * dt * (i*k)**2)
    def diffuse(self, z, x, x0, diff, dt):
        # imag = torch.tensor([-1], device=device)
        alpha = dt * diff * self.grid_size_x * self.grid_size_y * diff
        a = 0
        b = a + 1
        c = b + 1
        d = -b
        e = -c
        for k in range(20):
            x[b:d, b:d] = (x0[b:d, b:d] + alpha *
                           (x[a:e, b:d] + x[c:, b:d] + x[b:d, a:e] +
                            x[b:d, c:])) / (1 + 4 * alpha)

        x = self.set_bnd(self.grid_size_x, self.grid_size_y, z, x)
        return x, x0

    # Step 4
    # w4(k) = w3(k) - (i*k) * q
    # (i*k)**2 * q = (i*k) *w3(k)
    def project(self, u, v, u_prev, v_prev):
        h = 1. / ((self.grid_size_x + self.grid_size_y) / 2)
        a = 0
        b = a + 1
        c = b + 1
        d = -b
        e = -c
        v_prev[b:d, b:d] = -0.5 * h * ((u[c:, b:d] - u[:e, b:d]) + (v[b:d, c:] - v[b:d, :e]))
        u_prev.fill_(0.)
        u_prev = self.set_bnd(self.grid_size_x, self.grid_size_y, 0, u_prev)
        v_prev = self.set_bnd(self.grid_size_x, self.grid_size_y, 0, v_prev)

        for k in range(20):
            u_prev[b:d, b:d] = (v_prev[b:d, b:d] + u_prev[:e, b:d] +
                                u_prev[c:, b:d] + u_prev[b:d, :e] +
                                u_prev[b:d, c:]) / 4.
            u_prev = self.set_bnd(self.grid_size_x, self.grid_size_y, 0, u_prev)

        u[b:d, b:d] = u[b:d, b:d] - 0.5 * (u_prev[c:, b:d] - u_prev[:e, b:d]) / h
        v[b:d, b:d] = v[b:d, b:d] - 0.5 * (u_prev[b:d, c:] - u_prev[b:d, :e]) / h
        u = self.set_bnd(self.grid_size_x, self.grid_size_y, 1, u)
        v = self.set_bnd(self.grid_size_x, self.grid_size_y, 2, v)
        return u, v, u_prev, v_prev

    # Step 4.5
    # back transform from k space to x space
    def transform_to_x_space(self, w4_k):
        w4 = w4_k  # torch.fft.ifft2(w4_k)
        # TODO : write this method for comparision
        return w4

    def vel_step(self, fuel_density, oxidizer_density, product_density, u, v, u_prev, v_prev, viscosity, dt, step):
        u, u_prev = self.add_source_u(fuel_density, u, u_prev, dt, step)
        v, v_prev = self.add_source_v(fuel_density, v, v_prev, dt, step)
        u, u_prev = self.SWAP(u, u_prev)
        v, v_prev = self.SWAP(v, v_prev)
        u, u_prev = self.diffuse(1, u, u_prev, viscosity, dt)
        v, v_prev = self.diffuse(2, v, v_prev, viscosity, dt)
        u, v, u_prev, v_prev = self.project(u, v, u_prev, v_prev)
        u, u_prev = self.SWAP(u, u_prev)
        v, v_prev = self.SWAP(v, v_prev)
        u, u_prev = self.advect(1, u, u_prev, u_prev, v_prev, dt)
        v, v_prev = self.advect(2, v, v_prev, u_prev, v_prev, dt)
        u, v, u_prev, v_prev = self.project(u, v, u_prev, v_prev)
        return u, v, u_prev, v_prev, fuel_density, oxidizer_density, product_density

    def dens_step(self, fuel_density, fuel_density_prev, oxidizer_density, oxidizer_density_prev, product_density,
                  product_density_prev, u, v,
                  diff, dt, step):
        oxidizer_density, oxidizer_density_prev = self.add_oxidiser_density(oxidizer_density, oxidizer_density_prev, dt,
                                                                            step)
        fuel_density, fuel_density_prev = self.add_fuel_density(fuel_density, fuel_density_prev, dt, step)

        fuel_density, fuel_density_prev = self.SWAP(fuel_density, fuel_density_prev)
        oxidizer_density, oxidizer_density_prev = self.SWAP(oxidizer_density, oxidizer_density_prev)
        product_density, product_density_prev = self.SWAP(product_density, product_density_prev)

        fuel_density, fuel_density_prev = self.diffuse(0, fuel_density, fuel_density_prev, diff, dt)
        oxidizer_density, oxidizer_density_prev = self.diffuse(0, oxidizer_density, oxidizer_density_prev, diff, dt)
        product_density, product_density_prev = self.diffuse(0, product_density, product_density_prev, diff, dt)

        fuel_density, fuel_density_prev = self.SWAP(fuel_density, fuel_density_prev)
        oxidizer_density, oxidizer_density_prev = self.SWAP(oxidizer_density, oxidizer_density_prev)
        product_density, product_density_prev = self.SWAP(product_density, product_density_prev)

        fuel_density, fuel_density_prev = self.advect(0, fuel_density, fuel_density_prev, u, v, dt)
        oxidizer_density, oxidizer_density_prev = self.advect(0, oxidizer_density, oxidizer_density_prev, u, v, dt)
        product_density, product_density_prev = self.advect(0, product_density, product_density_prev, u, v, dt)

        return fuel_density, fuel_density_prev, oxidizer_density, oxidizer_density_prev, product_density, product_density_prev

    def pressure_poisson(self, p, poisson_vel_term, l2_target):
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

            p = self.set_bnd(self.grid_size_x, self.grid_size_y, 0, p)

            if n % 10 == 0:
                iter_diff = torch.sqrt(torch.sum((p - pn) ** 2) / torch.sum(pn ** 2))

            n += 1

        return p

    @staticmethod
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

    def update_grid(self, fuel_density, fuel_density_prev, oxidizer_density,
                    oxidizer_density_prev, product_density,
                    product_density_prev, u, u_prev, v, v_prev,
                    pressure, temperature, temperature_prev, mass_fuel, mass_oxidizer,
                    mass_product, poisson_v_term,
                    dt, viscosity, diff, step):
        temperature, temperature_prev = self.SWAP(temperature, temperature_prev)
        fuel_density, fuel_density_prev, oxidizer_density, oxidizer_density_prev, product_density, product_density_prev = \
            self.dens_step(fuel_density,
                           fuel_density_prev,
                           oxidizer_density,
                           oxidizer_density_prev,
                           product_density,
                           product_density_prev, u, v,
                           diff, dt, step)
        mass_fuel = fuel_density * self.grid_unit_volume
        mass_oxidizer = oxidizer_density * self.grid_unit_volume
        mass_product = product_density * self.grid_unit_volume
        u, v, u_prev, v_prev, fuel_density, oxidizer_density, product_density = self.vel_step(fuel_density,
                                                                                              oxidizer_density,
                                                                                              product_density, u, v,
                                                                                              u_prev,
                                                                                              v_prev, viscosity, dt,
                                                                                              step)

        u, v, fuel_density, oxidizer_density, product_density = self.combustion(fuel_density, oxidizer_density,
                                                                                product_density, u, v, temperature,
                                                                                step)
        u, v, oxidizer_density = self.evaporation_cooling(oxidizer_density, u, v)
        u, v, product_density = self.radiative_cooling(fuel_density, oxidizer_density, product_density, u, v,
                                                       temperature)
        u, v, product_density = self.radiative_heating(fuel_density, oxidizer_density, product_density, u, v,
                                                       temperature)

        velocity_magnitude = torch.sqrt(u ** 2 + v ** 2)
        poisson_v_term = self.poisson_velocity_term(poisson_v_term, fuel_density, oxidizer_density, product_density, dt,
                                                    u, v,
                                                    self.dx)
        pressure = self.pressure_poisson(pressure, velocity_magnitude, 0.1)
        temperature = self.velocity2temperature(velocity_magnitude, fuel_density, oxidizer_density, product_density,
                                                mass_fuel + mass_oxidizer + mass_product, self.degrees_of_freedom)

        rgb, alpha = self.temperature2rgb(temperature)

        # rgb = torch.cat([rgb , fuel_density.unsqueeze(2)],dim=2)
        return fuel_density, fuel_density_prev, \
            oxidizer_density, oxidizer_density_prev, \
            product_density, product_density_prev, \
            u, v, u_prev, v_prev, velocity_magnitude, pressure, \
            temperature, temperature_prev, \
            mass_fuel, mass_oxidizer, mass_product, \
            poisson_v_term, rgb, alpha

    def save_results(self, step, frame_skip=1, save_v=0, save_u=0, save_vu_mag=0, save_fuel=0, save_oxidizer=0,
                     save_product=0, save_pressure=0, save_temperature=0, save_rgb=0, save_alpha=0):
        if step % frame_skip == 0 or step == 0:
            if save_v == 1 and step == 0:
                self.v_result = torch.zeros_like(self.v, device=self.device)
            elif save_v == 1 and step != 0:
                self.v_result = torch.cat([self.v_result, self.v], dim=1)
            else:
                pass
            if save_u == 1 and step == 0:
                self.u_result = torch.zeros_like(self.u, device=self.device)
            elif save_u == 1 and step != 0:
                self.u_result = torch.cat([self.u_result, self.v], dim=1)
            else:
                pass
            if save_vu_mag == 1 and step == 0:
                self.uv_mag_result = torch.zeros_like(self.velocity_magnitude, device=self.device)
            elif save_vu_mag == 1 and step != 0:
                self.uv_mag_result = torch.cat([self.uv_mag_result, self.velocity_magnitude], dim=1)
            else:
                pass
            if save_fuel == 1 and step == 0:
                self.fuel_result = torch.zeros_like(self.fuel_density, device=self.device)
            elif save_fuel == 1 and step != 0:
                self.fuel_result = torch.cat([self.fuel_result, self.fuel_density], dim=1)
            else:
                pass
            if save_oxidizer == 1 and step == 0:
                self.oxidizer_result = torch.zeros_like(self.oxidizer_density, device=self.device)
            elif save_oxidizer == 1 and step != 0:
                self.oxidizer_result = torch.cat([self.oxidizer_result, self.oxidizer_density], dim=1)
            else:
                pass
            if save_product == 1 and step == 0:
                self.product_result = torch.zeros_like(self.product_density, device=self.device)
            elif save_product == 1 and step != 0:
                self.product_result = torch.cat([self.product_result, self.product_result], dim=1)
            else:
                pass
            if save_pressure == 1 and step == 0:
                self.pressure_result = torch.zeros_like(self.pressure, device=self.device)
            elif save_pressure == 1 and step != 0:
                self.pressure_result = torch.cat([self.pressure_result, self.pressure], dim=1)
            else:
                pass
            if save_temperature == 1 and step == 0:
                self.temperature_result = torch.zeros_like(self.temperature, device=self.device)
            elif save_temperature == 1 and step != 0:
                self.temperature_result = torch.cat([self.temperature_result, self.temperature], dim=1)
            else:
                pass
            if save_rgb == 1 and step == 0:
                self.rgb_result = torch.zeros_like(self.rgb, device=self.device)
            elif save_rgb == 1 and step != 0:
                self.rgb_result = torch.cat([self.rgb_result, self.rgb], dim=1)
            else:
                pass
            if save_alpha == 1 and step == 0:
                self.alpha_result = torch.zeros_like(self.alpha, device=self.device)
            elif save_alpha == 1 and step != 0:
                self.alpha_result = torch.cat([self.alpha_result, self.alpha], dim=1)
            else:
                pass

    def simulate(self, plot=0, save_animation=0, frame_skip=1, save_v=0, save_u=0, save_vu_mag=0, save_fuel=0,
                 save_oxidizer=0,
                 save_product=0, save_pressure=0, save_temperature=0, save_rgb=0, save_alpha=0):

        if plot == 1:
            # Create animation
            fig = plt.figure(figsize=(10, 6))
            grid = (2, 6)
            ax1 = plt.subplot2grid(grid, (0, 2))
            ax2 = plt.subplot2grid(grid, (0, 3))
            ax3 = plt.subplot2grid(grid, (0, 4))
            ax4 = plt.subplot2grid(grid, (0, 5))
            ax5 = plt.subplot2grid(grid, (1, 2))
            ax6 = plt.subplot2grid(grid, (1, 3))
            ax7 = plt.subplot2grid(grid, (1, 4))
            ax8 = plt.subplot2grid(grid, (1, 5))
            ax9 = plt.subplot2grid(grid, (0, 0), rowspan=2, colspan=2)
            for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]:
                ax.set_axis_off()

            font_title_size = 7
            ax1.set_title('Velocity field\n u component', size=font_title_size)
            ax2.set_title('Velocity field\n v component', size=font_title_size)
            ax3.set_title('Velocity\n Magnitude', size=font_title_size)
            ax4.set_title('Pressure\n Field', size=font_title_size)
            ax5.set_title('Fuel\n Density', size=font_title_size)
            ax6.set_title('Oxidizer\n Density', size=font_title_size)
            ax7.set_title('Product\n Densitty', size=font_title_size)
            ax8.set_title('Temperature \n Field (K)', size=font_title_size)
            ax9.set_title('RGB', size=font_title_size)
            ims = []

        for i in range(self.no_frames):
            self.save_results(i, save_v=save_v, save_u=save_u, save_vu_mag=save_vu_mag, save_fuel=save_fuel,
                              save_oxidizer=save_oxidizer,
                              save_product=save_product, save_pressure=save_pressure, save_temperature=save_temperature,
                              save_rgb=save_rgb, save_alpha=save_alpha)
            self.fuel_density, self.fuel_density_prev, \
                self.oxidizer_density, self.oxidizer_density_prev, \
                self.product_density, self.product_density_prev, \
                self.u, self.v, self.u_prev, self.v_prev, \
                self.pressure, self.velocity_magnitude, self.temperature, self.temperature_prev, \
                self.mass_fuel, self.mass_oxidizer, self.mass__product, \
                self.poisson_v_term, self.rgb, self.alpha = \
                self.update_grid(self.fuel_density, self.fuel_density_prev,
                                 self.oxidizer_density, self.oxidizer_density_prev,
                                 self.product_density, self.product_density_prev, self.u,
                                 self.u_prev, self.v,
                                 self.v_prev, self.pressure, self.temperature, self.temperature_prev,
                                 self.mass_fuel, self.mass_oxidizer, self.mass_product,
                                 self.poisson_v_term, self.dt,
                                 self.viscosity, self.diff, i)
            if plot == 1:
                if i % frame_skip == 0:
                    u_component = ax1.imshow(self.u.cpu().numpy(), animated=True)
                    v_component = ax2.imshow(self.v.cpu().numpy(), cmap='terrain', animated=True)
                    vel_mag = ax3.imshow(self.velocity_magnitude.cpu().numpy(), cmap='copper', animated=True)
                    pressure_field = ax4.imshow(self.pressure.cpu().numpy(), cmap='inferno', animated=True)
                    d = ax5.imshow(self.fuel_density.cpu().numpy(), cmap='hot', animated=True)
                    ox2 = ax6.imshow(self.oxidizer_density.cpu().numpy(), cmap='cool', animated=True)
                    combustion_products = ax7.imshow(self.product_density.cpu().numpy(), cmap='rainbow', animated=True)
                    temp = ax8.imshow((self.temperature.cpu().numpy()), cmap='plasma')
                    rgb = ax9.imshow((self.rgb.cpu().numpy()).astype(np.uint8), alpha=self.alpha.cpu().numpy())
                    ims.append(
                        [d, ox2, combustion_products, u_component, v_component, pressure_field, temp, vel_mag, rgb])
        if plot == 1:
            ani = animation.ArtistAnimation(fig, ims, interval=1, blit=True, repeat_delay=100)
        if save_animation == 1:
            ani.save("fixed_alpha.gif")
        if plot == 1:
            plt.show()

        torch.cuda.empty_cache()
        import sys

        sys.modules[__name__].__dict__.clear()
        import gc

        gc.collect()
