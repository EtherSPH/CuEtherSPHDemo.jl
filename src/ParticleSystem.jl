#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2024/08/29 19:28:52
  @ license: MIT
  @ description:
 =#

const kMaxNeighbourNumber::IntType = 100

@inline function capacityExpandPolicy(n::IntType)::IntType
    return n
end

mutable struct ParticleSystem
    n_particles_::IntType
    # * must have
    cu_position_::CuArray{RealType, 2, CUDA.DeviceMemory}
    cu_density_::CuArray{RealType, 1, CUDA.DeviceMemory}
    cu_mass_::CuArray{RealType, 1, CUDA.DeviceMemory}
    cu_type_::CuArray{IntType, 1, CUDA.DeviceMemory}
    cu_is_alive_::CuArray{BoolType, 1, CUDA.DeviceMemory} # true: alive, false: dead
    cu_cell_index_::CuArray{IntType, 1, CUDA.DeviceMemory} # which cell the particle belongs to
    # * neighbour information
    cu_neighbour_count_::CuArray{IntType, 1, CUDA.DeviceMemory}
    cu_neighbour_index_list_::CuArray{IntType, 2, CUDA.DeviceMemory}
    cu_neighbour_position_list_::CuArray{RealType, 3, CUDA.DeviceMemory}
    cu_neighbour_distance_list_::CuArray{RealType, 2, CUDA.DeviceMemory}
    # * additional property
    cu_pressure_::CuArray{RealType, 1, CUDA.DeviceMemory}
    cu_density_ratio_::CuArray{RealType, 1, CUDA.DeviceMemory}
    cu_velocity_::CuArray{RealType, 2, CUDA.DeviceMemory}
    cu_acceleration_::CuArray{RealType, 2, CUDA.DeviceMemory}
    cu_sound_speed_::CuArray{RealType, 1, CUDA.DeviceMemory}
    cu_viscosity_::CuArray{RealType, 1, CUDA.DeviceMemory}
    cu_particle_gap_::CuArray{RealType, 1, CUDA.DeviceMemory}
    # * sum kernel value
    cu_sum_kernel_weight_::CuArray{RealType, 1, CUDA.DeviceMemory}
    cu_sum_kernel_weighted_density_::CuArray{RealType, 1, CUDA.DeviceMemory}
    cu_sum_kernel_weighted_pressure_::CuArray{RealType, 1, CUDA.DeviceMemory}
end

@inline function ParticleSystem(
    n_particles::IntType;
    n_neighbours::IntType = kMaxNeighbourNumber,
    capacityPolicy::Function = capacityExpandPolicy,
)::ParticleSystem
    n_particles_capacity = capacityPolicy(n_particles)
    # * must have
    position = CUDA.fill(RealType(0), n_particles_capacity, 2) # i_particle, i_dim
    density = CUDA.fill(RealType(0), n_particles_capacity) # i_particle
    mass = CUDA.fill(RealType(0), n_particles_capacity) # i_particle
    type = CUDA.fill(IntType(0), n_particles_capacity) # i_particle
    is_alive = CUDA.fill(BoolType(false), n_particles_capacity) # i_particle
    cell_index = CUDA.fill(IntType(0), n_particles_capacity) # i_particle
    # * neighbour information
    neighbour_count = CUDA.fill(IntType(0), n_particles_capacity) # i_particle
    neighbour_index_list = CUDA.fill(IntType(0), n_particles_capacity, n_neighbours) # i_particle, i_neighbour
    neighbour_position_list = CUDA.fill(RealType(0), n_particles_capacity, n_neighbours, 2) # i_particle, i_neighbour, i_dim
    neighbour_distance_list = CUDA.fill(RealType(0), n_particles_capacity, n_neighbours) # i_particle, i_neighbour
    # * additional property
    pressure = CUDA.fill(RealType(0), n_particles_capacity) # i_particle
    density_ratio = CUDA.fill(RealType(0), n_particles_capacity) # i_particle
    velocity = CUDA.fill(RealType(0), n_particles_capacity, 2) # i_particle, i_dim
    acceleration = CUDA.fill(RealType(0), n_particles_capacity, 2) # i_particle, i_dim
    sound_speed = CUDA.fill(RealType(0), n_particles_capacity) # i_particle
    viscosity = CUDA.fill(RealType(0), n_particles_capacity) # i_particle
    particle_gap = CUDA.fill(RealType(0), n_particles_capacity) # i_particle
    # * sum kernel value
    sum_kernel_weight = CUDA.fill(RealType(0), n_particles_capacity) # i_particle
    sum_kernel_weighted_density = CUDA.fill(RealType(0), n_particles_capacity) # i_particle
    sum_kernel_weighted_pressure = CUDA.fill(RealType(0), n_particles_capacity) # i_particle
    return ParticleSystem(
        n_particles,
        # * must have
        position,
        density,
        mass,
        type,
        is_alive,
        cell_index,
        # * neighbour information
        neighbour_count,
        neighbour_index_list,
        neighbour_position_list,
        neighbour_distance_list,
        # * additional property
        pressure,
        density_ratio,
        velocity,
        acceleration,
        sound_speed,
        viscosity,
        particle_gap,
        # * sum kernel value
        sum_kernel_weight,
        sum_kernel_weighted_density,
        sum_kernel_weighted_pressure,
    )
end

@inline function Base.show(io::IO, ps::ParticleSystem)
    println(io, "ParticleSystem:")
    println(io, "  number of particles: $(ps.n_particles_)")
    println(io, "  number of alive particles: $(IntType(sum(ps.cu_is_alive_)))")
    return nothing
end
