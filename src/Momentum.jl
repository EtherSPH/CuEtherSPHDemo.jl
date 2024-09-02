#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2024/09/01 20:23:39
  @ license: MIT
  @ description:
 =#

@inline function harmonicMean(a::RealType, b::RealType)::RealType
    return 2 * a * b / (a + b)
end

const alpha::RealType = 0.5
const beta::RealType = 0.5

@inline function cuMomentum!(
    kernel::SPHKernel,
    n_particles::IntType,
    cu_is_alive,
    cu_neighbour_count,
    cu_neighbour_index_list,
    cu_neighbour_position_list,
    cu_neighbour_distance_list,
    cu_density,
    cu_mass,
    cu_type,
    cu_pressure,
    cu_velocity,
    cu_acceleration,
    cu_sound_speed,
    cu_viscosity,
    cu_particle_gap,
)::Nothing
    index = (blockIdx().x - IntType(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    for p in index:stride:n_particles
        @inbounds if p > n_particles
            break
        end
        @inbounds if cu_is_alive[p] == false || cu_neighbour_count[p] == IntType(0)
            continue
        end
        for i_neighbour in 1:cu_neighbour_count[p]
            @inbounds q = cu_neighbour_index_list[p, i_neighbour]
            if cu_is_alive[q] == true
                if cu_type[p] == FLUID_TAG && cu_type[q] == FLUID_TAG
                    @inbounds dr = cu_neighbour_distance_list[p, i_neighbour]
                    @inbounds dx = cu_neighbour_position_list[p, i_neighbour, 1]
                    @inbounds dy = cu_neighbour_position_list[p, i_neighbour, 2]
                    @inbounds mean_gap = RealType(0.5 * (cu_particle_gap[p] + cu_particle_gap[q]))
                    dw = kernelGradient(dr, kernel)
                    w = kernelValue(dr, kernel)
                    rw = kernelValue(mean_gap, kernel)
                    # * pressure force
                    @inbounds pressure_force =
                        cu_pressure[p] / (cu_density[p] * cu_density[p]) +
                        cu_pressure[q] / (cu_density[q] * cu_density[q])
                    pressure_force += abs(pressure_force) * w * RealType(0.01) / rw
                    @inbounds pressure_force *= -cu_mass[q] * dw / dr
                    @inbounds cu_acceleration[p, 1] += pressure_force * dx
                    @inbounds cu_acceleration[p, 2] += pressure_force * dy
                    # * viscosity force
                    @inbounds mean_viscosity = harmonicMean(cu_viscosity[p], cu_viscosity[q])
                    @inbounds viscosity_force =
                        2 * mean_viscosity * cu_mass[q] * dr * dw /
                        (cu_density[p] * cu_density[q] * (dr * dr + RealType(0.01) * mean_gap * mean_gap))
                    @inbounds du = cu_velocity[p, 1] - cu_velocity[q, 1]
                    @inbounds dv = cu_velocity[p, 2] - cu_velocity[q, 2]
                    @inbounds cu_acceleration[p, 1] += viscosity_force * du
                    @inbounds cu_acceleration[p, 2] += viscosity_force * dv
                    # * artificial viscosity
                    v_dot_x = du * dx + dv * dy
                    if v_dot_x < 0
                        @inbounds mean_rho = RealType(0.5 * (cu_density[p] + cu_density[q]))
                        @inbounds mean_sound_speed = harmonicMean(cu_sound_speed[p], cu_sound_speed[q])
                        phi = kernel.h_ * v_dot_x / (dr * dr + RealType(0.01) * mean_gap * mean_gap)
                        artificial_stress = RealType(-alpha * mean_sound_speed + beta * phi) * phi / mean_rho
                        @inbounds artificial_stress *= -cu_mass[q] * dw / (cu_density[p] * cu_density[q] * dr)
                        @inbounds cu_acceleration[p, 1] += artificial_stress * dx
                        @inbounds cu_acceleration[p, 2] += artificial_stress * dy
                    end
                elseif cu_type[p] == FLUID_TAG && cu_type[q] == WALL_TAG
                    @inbounds dr = cu_neighbour_distance_list[p, i_neighbour]
                    @inbounds dx = cu_neighbour_position_list[p, i_neighbour, 1]
                    @inbounds dy = cu_neighbour_position_list[p, i_neighbour, 2]
                    @inbounds mean_gap = RealType(0.5 * (cu_particle_gap[p] + cu_particle_gap[q]))
                    dw = kernelGradient(dr, kernel)
                    w = kernelValue(dr, kernel)
                    rw = kernelValue(mean_gap, kernel)
                    # * pressure force
                    @inbounds pressure_force =
                        2 * (cu_pressure[p] * cu_density[p] + cu_pressure[q] * cu_density[q]) /
                        ((cu_density[p] + cu_density[q]) * cu_density[p] * cu_density[q])
                    pressure_force += abs(pressure_force) * w * RealType(0.01) / rw
                    @inbounds pressure_force *= -cu_mass[q] * dw / dr
                    @inbounds cu_acceleration[p, 1] += pressure_force * dx
                    @inbounds cu_acceleration[p, 2] += pressure_force * dy
                    # * viscosity force
                    @inbounds mean_viscosity = harmonicMean(cu_viscosity[p], cu_viscosity[q])
                    @inbounds viscosity_force =
                        2 * mean_viscosity * cu_mass[q] * dr * dw /
                        (cu_density[p] * cu_density[q] * (dr * dr + RealType(0.01) * mean_gap * mean_gap))
                    @inbounds du = cu_velocity[p, 1] - cu_velocity[q, 1]
                    @inbounds dv = cu_velocity[p, 2] - cu_velocity[q, 2]
                    @inbounds cu_acceleration[p, 1] += viscosity_force * du
                    @inbounds cu_acceleration[p, 2] += viscosity_force * dv
                    # * artificial viscosity
                    v_dot_x = du * dx + dv * dy
                    if v_dot_x < 0
                        @inbounds mean_rho = RealType(0.5 * (cu_density[p] + cu_density[q]))
                        @inbounds mean_sound_speed = harmonicMean(cu_sound_speed[p], cu_sound_speed[q])
                        phi = kernel.h_ * v_dot_x / (dr * dr + RealType(0.01) * mean_gap * mean_gap)
                        artificial_stress = RealType(-alpha * mean_sound_speed + beta * phi) * phi / mean_rho
                        @inbounds artificial_stress *= -cu_mass[q] * dw / (cu_density[p] * cu_density[q] * dr)
                        @inbounds cu_acceleration[p, 1] += artificial_stress * dx
                        @inbounds cu_acceleration[p, 2] += artificial_stress * dy
                    end
                end
            end
        end
    end
    return nothing
end

@inline function momentum!(particle_system::ParticleSystem, kernel::SPHKernel; numthreads = 256)::Nothing
    numblocks = ceil(Int, particle_system.n_particles_ / numthreads)
    CUDA.@sync begin
        @cuda threads = numthreads blocks = numblocks cuMomentum!(
            kernel,
            particle_system.n_particles_,
            particle_system.cu_is_alive_,
            particle_system.cu_neighbour_count_,
            particle_system.cu_neighbour_index_list_,
            particle_system.cu_neighbour_position_list_,
            particle_system.cu_neighbour_distance_list_,
            particle_system.cu_density_,
            particle_system.cu_mass_,
            particle_system.cu_type_,
            particle_system.cu_pressure_,
            particle_system.cu_velocity_,
            particle_system.cu_acceleration_,
            particle_system.cu_sound_speed_,
            particle_system.cu_viscosity_,
            particle_system.cu_particle_gap_,
        )
    end
    return nothing
end
