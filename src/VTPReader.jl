#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2024/09/02 14:05:44
  @ license: MIT
  @ description:
 =#

@inline function readVTP(
    vtp_file_name::String;
    n_neighbours::IntType = kMaxNeighbourNumber,
    capacityPolicy::Function = capacityExpandPolicy,
)::ParticleSystem
    py"""
    import numpy as np
    import pyvista as pv

    def readVTP(vtp_file_name: str) -> tuple:
        vtp: pv.PolyData = pv.read(vtp_file_name)
        n_particles: int = vtp.n_points
        position: np.ndarray = vtp.points[:, 0:2]
        density: np.ndarray = vtp.point_data["Density"]
        mass: np.ndarray = vtp.point_data["Mass"]
        type_: np.ndarray = vtp.point_data["Type"]
        pressure: np.ndarray = vtp.point_data["Pressure"]
        velocity: np.ndarray = vtp.point_data["Velocity"][:, 0:2]
        sound_speed: np.ndarray = vtp.point_data["SoundSpeed"]
        viscosity: np.ndarray = vtp.point_data["Viscosity"]
        particle_gap: np.ndarray = vtp.point_data["Gap"]
        return (
            n_particles,
            position,
            density, mass, type_,
            pressure,
            velocity,
            sound_speed,
            viscosity,
            particle_gap,
        )
        pass
    """
    n_particles, position, density, mass, type_, pressure, velocity, sound_speed, viscosity, particle_gap =
        py"readVTP"(vtp_file_name)
    n_particles = IntType(n_particles)
    particle_system = ParticleSystem(n_particles; n_neighbours = n_neighbours, capacityPolicy = capacityPolicy)
    particle_system.cu_position_[1:n_particles, :] .= RealType.(position) |> CuArray
    particle_system.cu_density_[1:n_particles] .= RealType.(density) |> CuArray
    particle_system.cu_mass_[1:n_particles] .= RealType.(mass) |> CuArray
    particle_system.cu_type_[1:n_particles] .= IntType.(type_) |> CuArray
    particle_system.cu_is_alive_[1:n_particles] .= BoolType(true)
    particle_system.cu_pressure_[1:n_particles] .= RealType.(pressure) |> CuArray
    particle_system.cu_velocity_[1:n_particles, :] .= RealType.(velocity) |> CuArray
    particle_system.cu_sound_speed_[1:n_particles] .= RealType.(sound_speed) |> CuArray
    particle_system.cu_viscosity_[1:n_particles] .= RealType.(viscosity) |> CuArray
    particle_system.cu_particle_gap_[1:n_particles] .= RealType.(particle_gap) |> CuArray
    return particle_system
end
