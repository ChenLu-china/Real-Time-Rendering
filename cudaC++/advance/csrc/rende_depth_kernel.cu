#include <torch/extension.h>
#include <cstdint>
#include "include/render_util.cuh"
#include "include/cuda_util.cuh"
#include "include/packed_data_spec.cuh"
#include "include/random_util.cuh"

namespace{

const int WARP_SIZE = 32;

const int TRACE_RAY_CUDA_THREADS = 128;
const int TRACE_RAY_CUDA_RAYS_PER_BLOCK = TRACE_RAY_CUDA_THREADS / WARP_SIZE;

const int TRACE_RAY_BKWD_CUDA_THREADS = 128;
const int TRACE_RAY_BKWD_CUDA_RAYS_PER_BLOCK = TRACE_RAY_BKWD_CUDA_THREADS / WARP_SIZE;

const int MIN_BLOCKS_PER_SM = 8;

const float URF_SIGMA_SCALE_FACTOR = 3.0;

typedef cub::WarpReduce<float> WarpReducef;

namespace device{

__device__ __inline__ void trace_ray_depth(
    const PackedSparseGridSpec& __restrict__ grid,
    SingleRaySpec& __restrict__ ray,
    const RenderOptions& __restrict__ opt,
    uint32_t lane_id,
    float* __restrict__ sphfunc_val,
    WarpReducef::TempStorage& __restrict__ temp_storage,
    float* __restrict__ rgb_out,
    float* __restrict__ depth_out,
    float* __restrict__ out_log_transmit){
    
    const uint32_t lane_colorgrp_id = lane_id % grid.basis_dim; // 0-26, because sh_data has 27 channel
    const uint32_t lane_colorgrp = lane_id / grid.basis_dim; // 0 , 1 , 2 because rgb has three channel

    if(ray.tmin> ray.tmax){
        rgb_out[lane_colorgrp] = (grid.background_nlayers == 0) ? opt.background_brightness : 0.f;
        *depth_out = 0.f;
        if(out_log_transmit != nullptr){
            *out_log_transmit = 0.f;
        }
        return;
    }

    float t = ray.tmin;
    float outv = 0.f;
    float depth_outv = 0.f;
    
    float log_transmit = 0.f;
    
    while(t <= ray.tmax){
#pragma unroll 3
        for(int j = 0; j < 3; ++j){
            ray.pos[j] = fmaf(t, ray.dir[j], ray.origin[j]);
            ray.pos[j] = min(max(ray.pos[j], 0.f), grid.size[j] - 1.f);

            ray.l[j] = min(static_cast<int32_t>(ray.pos[j]), grid.size[j] - 2);
            ray.pos[j] -= static_cast<float>(ray.l[j]);
        }
        const float skip = compute_skip_dist(ray, grid.links, grid.stride_x, grid.size[2], 0);
        if(skip >= opt.step_size){
            t += ceilf(skip / opt.step_size) * opt.step_size;
            continue;
        }
        float sigma = trilerp_cuvol_one(
            grid.links,
            grid.density_data,
            grid.stride_x,
            grid.size[2],
            1,
            ray.l,
            ray.pos,
            0);

        if(opt.last_sample_opaque && t + opt.step_size > ray.tmax){
            ray.world_step = 1e9;
        }
        
        if(sigma > opt.sigma_thresh){
            float lane_color = trilerp_cuvol_one(
                                grid.links,
                                grid.sh_data,
                                grid.stride_x,
                                grid.size[2],
                                grid.sh_data_dim,
                                ray.l, ray.pos, lane_id);
            lane_color *= sphfunc_val[lane_colorgrp_id];

            const float pcnt = ray.world_step * sigma;
            const float weight = _EXP(log_transmit) * (1.f - _EXP(-pcnt));
            log_transmit -= pcnt;

            float lane_color_total = WarpReducef(temp_storage).HeadSegmentedSum(
                                                        lane_color, lane_colorgrp_id == 0);
            outv += weight * fmaxf(lane_color_total + 0.5f, 0.f);
            depth_outv += weight * (t / opt.step_size) * ray.world_step;
            
            if(_EXP(log_transmit) < opt.stop_thresh){
                log_transmit = -1e3f;
                break;
            }
        }
        t += opt.step_size;
    }
    
    if (grid.background_nlayers == 0) {
        outv += _EXP(log_transmit) * opt.background_brightness;
    }
    if (lane_colorgrp_id == 0) {
        *depth_out = depth_outv; // we only save this ones
        if (out_log_transmit != nullptr) {
            *out_log_transmit = log_transmit;
        }
        rgb_out[lane_colorgrp] = outv;
    }
}

__device__ __inline__ void trace_ray_depth_backward(
    const PackedSparseGridSpec& __restrict__ grid,
    const float* __restrict__ grad_output,
    const float* __restrict__ color_cache,
    const float* __restrict__ depth_output,
    const float* __restrict__ depth_cache,
    SingleRaySpec& __restrict__ ray,
    const RenderOptions& __restrict__ opt,
    uint32_t lane_id,
    const float* __restrict__ sphfunc_val,
    float* __restrict__ grad_sphfunc_val,
    WarpReducef::TempStorage& __restrict__ temp_storage,
    float log_transmit_in,
    float beta_loss,
    float sparsity_loss,
    PackedGridOutputGrads& __restrict__ grads,
    float* __restrict__ accum_out,
    float* __restrict__ log_transmit_out,
    const Normal& __restrict__ target_distribution){
    
    /*
        因为是并行运算所以逻辑比较混乱
        对于每个光线我们是从
    */
    const uint32_t lane_colorgrp_id = lane_id % grid.basis_dim;
    const uint32_t lane_colorgrp = lane_id / grid.basis_dim;
    // leader_mask 是对线程的一个掩码
    const uint32_t leader_mask = 1U | (1U << grid.basis_dim) | (1U << (2 * grid.basis_dim));


    
    float accum = fmaf(color_cache[0], grad_output[0], 
                        fmaf(color_cache[1], grad_output[1],
                            color_cache[2] * grad_output[2]));

    if (beta_loss > 0.f) {
        const float transmit_in = _EXP(log_transmit_in);
        beta_loss *= (1 - transmit_in / (1 - transmit_in + 1e-3)); // d beta_loss / d log_transmit_in
        accum += beta_loss;
        // Interesting how this loss turns out, kinda nice?
    }

    if (ray.tmin > ray.tmax) {
        if (accum_out != nullptr) { *accum_out = accum; }
        if (log_transmit_out != nullptr) { *log_transmit_out = 0.f; }
        // printf("accum_end_fg_fast=%f\n", accum);
        return;
    }
    float t = ray.tmin;
    const float gout = grad_output[lane_colorgrp];

    float log_transmit = 0.f;

    while (t <= ray.tmax) {
#pragma unroll 3
        for (int j = 0; j < 3; ++j) {
            ray.pos[j] = fmaf(t, ray.dir[j], ray.origin[j]);
            ray.pos[j] = min(max(ray.pos[j], 0.f), grid.size[j] - 1.f);
            ray.l[j] = min(static_cast<int32_t>(ray.pos[j]), grid.size[j] - 2);
            ray.pos[j] -= static_cast<float>(ray.l[j]);
        }
        const float skip = compute_skip_dist(ray,
                       grid.links, grid.stride_x,
                       grid.size[2], 0);
        if (skip >= opt.step_size) {
            // For consistency, we skip the by step size
            t += ceilf(skip / opt.step_size) * opt.step_size;
            continue;
        }

        float sigma = trilerp_cuvol_one(
                grid.links,
                grid.density_data,
                grid.stride_x,
                grid.size[2],
                1,
                ray.l, ray.pos,
                0);
        
        if (opt.last_sample_opaque && t + opt.step_size > ray.tmax) {
            ray.world_step = 1e9;
        }

        // 计算depth
        if(depth_cache <= t)

        // if (opt.randomize && opt.random_sigma_std > 0.0) sigma += ray.rng.randn() * opt.random_sigma_std;
        if (sigma > opt.sigma_thresh) {
            float lane_color = trilerp_cuvol_one(
                            grid.links,
                            grid.sh_data,
                            grid.stride_x,
                            grid.size[2],
                            grid.sh_data_dim,
                            ray.l, ray.pos, lane_id);
            float weighted_lane_color = lane_color * sphfunc_val[lane_colorgrp_id];

            const float pcnt = ray.world_step * sigma;
            const float weight = _EXP(log_transmit) * (1.f - _EXP(-pcnt));
            log_transmit -= pcnt;

            const float lane_color_total = WarpReducef(temp_storage).HeadSegmentedSum(
                                           weighted_lane_color, lane_colorgrp_id == 0) + 0.5f;
            float total_color = fmaxf(lane_color_total, 0.f);
            float color_in_01 = total_color == lane_color_total;
            total_color *= gout; // Clamp to [+0, infty)

            float total_color_c1 = __shfl_sync(leader_mask, total_color, grid.basis_dim);
            total_color += __shfl_sync(leader_mask, total_color, 2 * grid.basis_dim);
            total_color += total_color_c1;

            color_in_01 = __shfl_sync((1U << grid.sh_data_dim) - 1, color_in_01, lane_colorgrp * grid.basis_dim);
            const float grad_common = weight * color_in_01 * gout;
            const float curr_grad_color = sphfunc_val[lane_colorgrp_id] * grad_common;

            if (grid.basis_type != BASIS_TYPE_SH) {
                float curr_grad_sphfunc = lane_color * grad_common;
                const float curr_grad_up2 = __shfl_down_sync((1U << grid.sh_data_dim) - 1,
                        curr_grad_sphfunc, 2 * grid.basis_dim);
                curr_grad_sphfunc += __shfl_down_sync((1U << grid.sh_data_dim) - 1,
                        curr_grad_sphfunc, grid.basis_dim);
                curr_grad_sphfunc += curr_grad_up2;
                if (lane_id < grid.basis_dim) {
                    grad_sphfunc_val[lane_id] += curr_grad_sphfunc;
                }
            }

            accum -= weight * total_color;
            float curr_grad_sigma = ray.world_step * (
                    total_color * _EXP(log_transmit) - accum);
            
            // one of sigma loss
            if (sparsity_loss > 0.f) {
                    // Cauchy version (from SNeRG)
                    curr_grad_sigma += sparsity_loss * (4 * sigma / (1 + 2 * (sigma * sigma)));

                    // Alphs version (from PlenOctrees)
                    // curr_grad_sigma += sparsity_loss * _EXP(-pcnt) * ray.world_step;
            }
            // use gt depth to supervised depth
        }
    }
}


__launch_bounds__(TRACE_RAY_CUDA_THREADS, MIN_BLOCKS_PER_SM)
__global__ void trace_depth_kernel(
        PackedSparseGridSpec grid,
        PackedRaysSpec rays,
        RenderOptions opt,
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> rgb_out,
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> depth_out,
        float* __restrict__ log_transmit_out = nullptr){
    
    CUDA_GET_THREAD_ID(tid, int(rays.origins.size(0)) * WARP_SIZE);

    const int ray_id = tid >> 5;
    const int ray_blk_id = threadIdx.x >> 5;
    const int warp_thread_id = threadIdx.x & 0x1F;

    if(warp_thread_id > grid.sh_data_dim)
        return;
    
    __shared__ Normal target_distribution;
    __shared__ float sphfunc_val[TRACE_RAY_CUDA_RAYS_PER_BLOCK][9];
    __shared__ SingleRaySpec ray_specs[TRACE_RAY_CUDA_RAYS_PER_BLOCK];
    __shared__ typename WarpReducef::TempStorage temp_storage[
        TRACE_RAY_CUDA_RAYS_PER_BLOCK];
    
    target_distribution.set(0.0, opt.sigma / URF_SIGMA_SCALE_FACTOR);

    ray_specs[ray_blk_id].set(rays.origins[ray_id].data(), rays.dirs[ray_id].data());

    
    // float d = target_distribution.log_prob(0.0);
    // printf("Gaussian Distribution value is %f\n", d);

    calc_sphfunc(grid, warp_thread_id, ray_id, ray_specs[ray_blk_id].dir, sphfunc_val[ray_blk_id]);

    ray_find_bounds(ray_specs[ray_blk_id], grid, opt, ray_id);  // find bounds for each rays
    
    // 因为定义了共享内存变量， 固高并发的线程之间会有竞争， 所以使用__synthreads进行同步
    __syncwarp((1U << grid.sh_data_dim) - 1);  // (1U means unsigned value 1) is shift to the left by x bits.

    trace_ray_depth(
        grid,
        ray_specs[ray_blk_id],
        opt,
        warp_thread_id,
        sphfunc_val[ray_blk_id],
        temp_storage[ray_blk_id],
        rgb_out[ray_id].data(),
        depth_out[ray_id].data(),
        log_transmit_out == nullptr ? nullptr : log_transmit_out + ray_id);
}

__launch_bounds__(TRACE_RAY_CUDA_THREADS, MIN_BLOCKS_PER_SM)
__global__ void render_ray_backward_kernel(
    PackedSparseGridSpec grid,
    // rgb gt
    const float* __restrict__ grad_output,
    //rgb output
    const float* __restrict__ color_cache,
    const float* __restrict__ depth_output,
    const float* __restrict__ depth_cache,
    PackedRaysSpec rays,
    RenderOptions opt,
    bool grad_out_is_rgb,
    const float* __restrict__ log_transmit_in,
    float beta_loss,
    float sparsity_loss,
    PackedGridOutputGrads grads,
    float* __restrict__ accum_out = nullptr,
    float* __restrict__ log_transmit_out = nullptr
){
    CUDA_GET_THREAD_ID(tid, int(rays.origins.size(0)) * WARP_SIZE);
    const int ray_id = tid >> 5;
    const int ray_blk_id = threadIdx.x >> 5;
    const int warp_thread_id = threadIdx.x & 0x1F;

    if(warp_thread_id >= grid.sh_data_dim)
        return;

    __shared__ Normal target_distribution;
    __shared__ float sphfunc_val[TRACE_RAY_BKWD_CUDA_RAYS_PER_BLOCK][9];
    __shared__ float grad_sphfunc_val[TRACE_RAY_CUDA_RAYS_PER_BLOCK][9];
    __shared__ SingleRaySpec ray_spec[TRACE_RAY_BKWD_CUDA_RAYS_PER_BLOCK];
    __shared__ typename WarpReducef::TempStorage temp_storage[TRACE_RAY_CUDA_RAYS_PER_BLOCK];


    // Implement Urban NeRF Gaussian Distribution
    target_distribution.set(0.0, opt.sigma / URF_SIGMA_SCALE_FACTOR);
    ray_spec[ray_blk_id].set(rays.origins.data(), rays.dirs[ray_id].data());

    const float vdir[3] = {ray_spec[ray_blk_id].dir[0],
                    ray_spec[ray_blk_id].dir[1],
                    ray_spec[ray_blk_id].dir[2]};
    if(warp_thread_id < grid.basis_dim){
        grad_sphfunc_val[ray_blk_id][warp_thread_id] = 0.f;
    }
    calc_sphfunc(grid, warp_thread_id, 
                ray_id, vdir, 
                sphfunc_val[ray_blk_id]);
    if(warp_thread_id == 0){
        ray_find_bounds(ray_spec[ray_blk_id], grid, opt, ray_id);
    }
    
    float grad_out[3];
    if(grad_out_is_rgb){
        const float norm_factor = 2.f / (3 * int(rays.origins.size(0)));
#pragma unroll 3
        for(int i = 0; i < 3; ++i){
            const float resid = color_cache[ray_id * 3 + i] - grad_output[ray_id * 3 + i];
            grad_out[i] = resid * norm_factor;
        }
    }  else{
#pragma unroll 3
        for (int i = 0; i < 3; ++i){
            grad_out[i] = grad_output[ray_id * 3 + i];
        }
    }

    // expect depth grad
    float depth_grad_out;
    const float depth_norm_factor = 2.f / (int(rays.origins.size(0)));
    const float depth_resid = depth_cache[ray_id] - grad_output[ray_id];
    depth_grad_out = depth_resid * depth_norm_factor;

    __syncwarp((1U << grid.sh_data_dim) - 1);

    // 
    trace_ray_depth_backward(
        grid,
        grad_out,
        color_cache + ray_id * 3,
        &depth_grad_out,
        depth_cache + ray_id,
        ray_spec[ray_blk_id],
        opt,
        warp_thread_id,
        sphfunc_val[ray_blk_id],
        grad_sphfunc_val[ray_blk_id],
        temp_storage[ray_blk_id],
        log_transmit_in == nullptr ? 0.f : log_transmit_in[ray_id],
        beta_loss,
        sparsity_loss,
        grads,
        accum_out == nullptr ? nullptr : accum_out + ray_id,
        log_transmit_out == nullptr ? nullptr : log_transmit_out + ray_id,
        target_distribution);
}
} // namespace device    
} // namespace


torch::Tensor _get_empty_1d(const torch::Tensor origins){
    auto options = torch::TensorOptions()
                   .dtype(origins.dtype())
                   .layout(torch::kStrided)
                   .device(origins.device())
                   .requires_grad(false);
    return torch::empty({origins.size(0)}, options);
}


// define myself radar render function
void volume_render_optimize_fused(
    SparseGridSpec& grid,
    RaysSpec& rays,
    RenderOptions& opt,
    torch::Tensor rgb_gt,
    torch::Tensor rgb_out,
    torch::Tensor depth_gt,
    torch::Tensor depth_out,
    float denstiy_loss,
    float penalize_loss,
    float beta_loss,
    float sparsity_loss,
    GridOutputGrads& grads){
    
    DEVICE_GUARD(grid.sh_data);
    CHECK_INPUT(rgb_gt);
    CHECK_INPUT(rgb_out);
    CHECK_INPUT(depth_gt);
    CHECK_INPUT(depth_out);
    grid.check();
    rays.check();
    grads.check();    
    
    const auto Q = rays.origins.size(0);

    // if we input rays from our selection that means the inputs of this part we ensure that rays have accurate depths
    // we do not need to save log_transmit
    bool use_background = grid.background_links.defined() &&
                          grid.background_links.size(0) > 0;
    bool need_log_transmit = use_background || beta_loss > 0.f;
    
    torch::Tensor log_transmit, accum; // log_transmit to judge that the rays has intersection with forground objects.
    if (need_log_transmit) {
        log_transmit = _get_empty_1d(rays.origins);
    }
    if (use_background) {
        accum = _get_empty_1d(rays.origins);
    }
    
    bool need_log_weights = penalize_loss > 0.f;
    torch::Tensor log_weights; // I want to use weights to constraint depth
    if (need_log_weights){
        log_weights = _get_empty_1d(rays.origins);
    }
    
    {   
        //
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q * WARP_SIZE, TRACE_RAY_CUDA_THREADS);
        printf("We went there %d \n", blocks);
        device::trace_depth_kernel<<<blocks, TRACE_RAY_CUDA_THREADS>>>(
                grid, rays, opt,
                rgb_out.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                depth_out.packed_accessor32<float, 2, torch::RestrictPtrTraits>(), 
                need_log_transmit ? log_transmit.data_ptr<float>() : nullptr);
    }
    

    {
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q * WARP_SIZE, TRACE_RAY_BKWD_CUDA_THREADS);
        device::render_ray_backward_kernel<<<blocks, TRACE_RAY_BKWD_CUDA_THREADS>>>(
                grid,
                // grad_output
                rgb_gt.data_ptr<float>(),
                rgb_out.data_ptr<float>(),
                depth_gt.data_ptr<float>(),
                depth_out.data_ptr<float>(),
                rays, opt,
                true,
                beta_loss > 0.f ? log_transmit.data_ptr<float>() : nullptr,
                beta_loss / Q,
                sparsity_loss,
                grads,
                use_background ? accum.data_ptr<float>() : nullptr,
                nullptr
                );
    }

    
}