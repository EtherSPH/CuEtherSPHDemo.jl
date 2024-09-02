#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2024/09/01 16:07:48
  @ license: MIT
  @ description:
 =#

@inline function cuContinuityAndPressureExtrapolation!(
    kernel::SPHKernel,
    n_particles::IntType,
    body_force_x::RealType,
    body_force_y::RealType,
    cu_is_alive,
    cu_neighbour_count,
    cu_neighbour_index_list,
    cu_neighbour_position_list,
    cu_neighbour_distance_list,
    cu_density,
    cu_mass,
    cu_type,
    cu_pressure,
    cu_density_ratio,
    cu_velocity,
    cu_sum_kernel_weight,
    cu_sum_kernel_weighted_pressure,
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
                    @inbounds v_dot_x =
                        cu_neighbour_position_list[p, i_neighbour, 1] * (cu_velocity[p, 1] - cu_velocity[q, 1]) +
                        cu_neighbour_position_list[p, i_neighbour, 2] * (cu_velocity[p, 2] - cu_velocity[q, 2])
                    @inbounds dr = cu_neighbour_distance_list[p, i_neighbour]
                    dw = kernelGradient(dr, kernel)
                    @inbounds cu_density_ratio[p] += cu_mass[q] * v_dot_x * dw / dr
                elseif cu_type[p] == FLUID_TAG && cu_type[q] == WALL_TAG
                    @inbounds v_dot_x =
                        cu_neighbour_position_list[p, i_neighbour, 1] * (cu_velocity[p, 1] - cu_velocity[q, 1]) +
                        cu_neighbour_position_list[p, i_neighbour, 2] * (cu_velocity[p, 2] - cu_velocity[q, 2])
                    @inbounds dr = cu_neighbour_distance_list[p, i_neighbour]
                    dw = kernelGradient(dr, kernel)
                    @inbounds cu_density_ratio[p] += cu_density[p] * cu_mass[q] * v_dot_x * dw / (dr * cu_density[q])
                elseif cu_type[p] == WALL_TAG && cu_type[q] == FLUID_TAG
                    @inbounds w = kernelValue(cu_neighbour_distance_list[p, i_neighbour], kernel)
                    @inbounds kernel_weight = w * cu_mass[q] / cu_density[q]
                    @inbounds cu_sum_kernel_weight[p] += kernel_weight
                    @inbounds body_force_dot_x =
                        cu_neighbour_position_list[p, i_neighbour, 1] * body_force_x +
                        cu_neighbour_position_list[p, i_neighbour, 2] * body_force_y
                    pressure = max(cu_pressure[q], RealType(0)) + cu_density[q] * max(body_force_dot_x, RealType(0))
                    @inbounds cu_sum_kernel_weighted_pressure[p] += kernel_weight * pressure
                end
            end
        end
    end
    return nothing
end

@inline function continuityAndPressureExtrapolation!(
    particle_system::ParticleSystem,
    kernel::SPHKernel,
    body_force_x::RealType,
    body_force_y::RealType;
    numthreads = 256,
)::Nothing
    numblocks = ceil(Int, particle_system.n_particles_ / numthreads)
    CUDA.@sync begin
        @cuda threads = numthreads blocks = numblocks cuContinuityAndPressureExtrapolation!(
            kernel,
            particle_system.n_particles_,
            body_force_x,
            body_force_y,
            particle_system.cu_is_alive_,
            particle_system.cu_neighbour_count_,
            particle_system.cu_neighbour_index_list_,
            particle_system.cu_neighbour_position_list_,
            particle_system.cu_neighbour_distance_list_,
            particle_system.cu_density_,
            particle_system.cu_mass_,
            particle_system.cu_type_,
            particle_system.cu_pressure_,
            particle_system.cu_density_ratio_,
            particle_system.cu_velocity_,
            particle_system.cu_sum_kernel_weight_,
            particle_system.cu_sum_kernel_weighted_pressure_,
        )
    end
    return nothing
end
