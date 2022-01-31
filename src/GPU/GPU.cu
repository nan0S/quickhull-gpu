#include "GPU.cuh"

#include <cstdio>
#include <iostream>

#include <GL/glew.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/zip_function.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/extrema.h>
#include <thrust/partition.h>
#include <thrust/random.h>

#include "GPU/Error.h"
#include "Graphics/Error.h"
#include "Utils/Timer.h"
#include "Utils/Log.h"

namespace GPU
{
   /* macros */
   #define PI 3.14159265358f
   #define NTHREADS 1024
   #define NPERTHREAD 1024

   #define SEND_TO_GPU(symbol, expr) { \
               auto v = (expr); \
               cudaCall(cudaMemcpyToSymbol(symbol, &v, sizeof(v), 0, \
                                           cudaMemcpyHostToDevice)); }

   /* struct-kernels */
   struct is_above_line
   {
      __device__
      bool operator()(float x, float y) const;
   };

   struct calc_first_pts
   {
      __device__
      void operator()(int head, int key, int index) const;
   };

   struct calc_line_dist
   {
      __device__
      float operator()(float x, float y, int key, int hull_count) const;
   };

   struct update_heads
   {
      __device__
      void operator()(int index) const;
   };

   struct calc_outerior
   {
      __device__
      bool operator()(float x, float y, int key, int head, int hull_count) const;
   };

   struct is_on_hull
   {
      __device__
      bool operator()(int index, int hull_count) const;
   };

   /* structs */
   struct Memory
   {
      bool is_host_mem;
      GLuint gl_buffer;
      cudaGraphicsResource_t resource;
      void* d_buffer;
      float* h_buffer;
   };

   /* forward declarations */
   __global__
   void generatePoints(float* x, float* y, const int N);
   __device__
   float cross(float ux, float uy, float vx, float vy);
   size_t getCudaMemoryNeeded(int n);

   /* GPU variables */
   __constant__ int d_seed;
   __constant__ unsigned long long d_offset;
   __constant__ float d_r_min;
   __constant__ float d_r_diff;

   __constant__ float* d_x;
   __constant__ float* d_y;
   __constant__ int* d_head;
   __constant__ int* d_first_pts;
   __constant__ int* d_flag;

   __constant__ float d_left_x;
   __constant__ float d_left_y;
   __constant__ float d_right_x;
   __constant__ float d_right_y;

   /* CPU variables */
   Memory mem;

   void init(Config* config,
             const std::vector<int>& n_points,
             GLuint gl_buffer)
   {
      float r_min = 0.f, r_max = 1.f;
      switch (config->dataset_type)
      {
         case DatasetType::DISC:
            r_min = 0.f; r_max = 1.f;
            break;
         case DatasetType::RING:
            r_min = 0.9f; r_max = 1.f;
            break;
         case DatasetType::CIRCLE:
            r_min = 1.f; r_max = 1.f;
            break;
         default:
            assert(false);
      }
      float r_diff = r_max - r_min;
      cudaCall(cudaMemcpyToSymbol(d_seed, &config->seed, sizeof(int)));
      cudaCall(cudaMemcpyToSymbol(d_r_min, &r_min, sizeof(float)));
      cudaCall(cudaMemcpyToSymbol(d_r_diff, &r_diff, sizeof(float)));

      mem.is_host_mem = config->is_host_mem;

      int max_n = *thrust::max_element(n_points.begin(), n_points.end());
      size_t cuda_needed = getCudaMemoryNeeded(max_n);
      if (!mem.is_host_mem)
      {
         // Allocate OpenGL buffer and prepare to map it into CUDA.
         glCall(glBufferData(GL_ARRAY_BUFFER, cuda_needed, NULL, GL_STATIC_DRAW));
         cudaCall(cudaGraphicsGLRegisterBuffer(&mem.resource, gl_buffer,
                                               cudaGraphicsMapFlagsWriteDiscard));
      }
      else
      {
         // Allocate OpenGL, CUDA and host buffers. CUDA->host->OpenGL->draw.
         size_t bytes = 2 * max_n * sizeof(float);
         glCall(glBufferData(GL_ARRAY_BUFFER, bytes, NULL, GL_STATIC_DRAW));
         cudaCall(cudaMalloc(&mem.d_buffer, cuda_needed));
         cudaCall(cudaMallocHost(&mem.h_buffer, bytes));
      }

      glCall(glVertexAttribPointer(0, 1, GL_FLOAT, GL_FALSE, 0, 0));
   }

   int calculate(int n)
   {
      print("\nRunning GPU for ", n, " points.");

      // Initialize pointers to previously allocated memory.
      size_t cuda_needed = getCudaMemoryNeeded(n);
      if (!mem.is_host_mem)
      {
         size_t size = 0;
         cudaCall(cudaGraphicsMapResources(1, &mem.resource));
         cudaCall(cudaGraphicsResourceGetMappedPointer(&mem.d_buffer, &size, mem.resource));
         assert(size >= cuda_needed);
      }

      thrust::device_ptr<float> x(reinterpret_cast<float*>(mem.d_buffer));
      thrust::device_ptr<float> y(reinterpret_cast<float*>(x.get() + n));
      thrust::device_ptr<int> head(reinterpret_cast<int*>(y.get() + n));
      thrust::device_ptr<int> keys(reinterpret_cast<int*>(head.get() + n));
      thrust::device_ptr<int> first_pts(reinterpret_cast<int*>(keys.get() + n));
      thrust::device_ptr<int> flag(reinterpret_cast<int*>(first_pts.get() + n));
      thrust::device_ptr<float> dist(reinterpret_cast<float*>(flag.get() + n));

      SEND_TO_GPU(d_x, x.get());
      SEND_TO_GPU(d_y, y.get());
      SEND_TO_GPU(d_head, head.get());
      SEND_TO_GPU(d_first_pts, first_pts.get());
      SEND_TO_GPU(d_flag, flag.get());

      // Generate points.
      {
         static unsigned long long offset = 0;
         auto iceil = [](int x, int d) { return (x + d - 1) / d; };
         const int spawn_total = iceil(n, NPERTHREAD);
         const int nblocks = iceil(spawn_total, NTHREADS);
         SEND_TO_GPU(d_offset, offset);
         generatePoints<<<nblocks, NTHREADS>>>(x.get(), y.get(), n);
         offset += n;
      }

      Timer timer("QuickHull");

      // Find leftmost and rightmost points.
      auto it = thrust::minmax_element(thrust::make_zip_iterator(x, y),
                                       thrust::make_zip_iterator(x+n, y+n));
      auto it_left = it.first.get_iterator_tuple();
      auto it_right = it.second.get_iterator_tuple();
      cudaCall(cudaMemcpyToSymbol(d_left_x, it_left.get<0>().get(),
                                  sizeof(float), 0, cudaMemcpyDeviceToDevice));
      cudaCall(cudaMemcpyToSymbol(d_left_y, it_left.get<1>().get(),
                                  sizeof(float), 0, cudaMemcpyDeviceToDevice));
      cudaCall(cudaMemcpyToSymbol(d_right_x, it_right.get<0>().get(),
                                  sizeof(float), 0, cudaMemcpyDeviceToDevice));
      cudaCall(cudaMemcpyToSymbol(d_right_y, it_right.get<1>().get(),
                                  sizeof(float), 0, cudaMemcpyDeviceToDevice));
      int left_idx = static_cast<int>(it_left.get<0>() - x);
      int right_idx = static_cast<int>(it_right.get<0>() - x);

      // Partition into lower and upper parts.
      auto pivot = thrust::partition(thrust::make_zip_iterator(x, y),
                                     thrust::make_zip_iterator(x+n, y+n),
                                     thrust::make_zip_function(is_above_line{}));
      int pivot_idx = static_cast<int>(pivot.get_iterator_tuple().get<0>() - x);

      // Sort points in lower and upper parts.
      thrust::sort(thrust::make_zip_iterator(x, y), pivot,
                   thrust::greater<>());
      thrust::sort(pivot, thrust::make_zip_iterator(x+n, y+n));

      // Initialize head.
      cudaCall(cudaMemset(head.get(), 0, n * sizeof(int)));
      head[0] = 1;
      head[pivot.get_iterator_tuple().get<0>() - x] = 1;

      // Prepare variables.
      const int N = n;
      int hull_count = 0;
      int last_hull_count = 0;
      auto diter = thrust::make_discard_iterator();

      while (hull_count < n)
      {
         // Calculate keys from head.
         thrust::device_ptr<int> end = thrust::inclusive_scan(head, head+n, keys);
         hull_count = *(end - 1);
         // Line distance calculation ensured that segment borders will not
         // be selected as the farthest point in the segment (unless there
         // aren't anymore points in the segment). However if there still is
         // some precision-related issue, then this check is a guard from
         // an infinite loop. It should be always false, however I leave it
         // just in case (hull will be correct with respect to float::eps).
         if (hull_count == last_hull_count)
            break;
         last_hull_count = hull_count;
         thrust::for_each_n(keys, n, thrust::placeholders::_1 -= 1);

         // Calculate first_pts from keys and head.
         thrust::counting_iterator<int> iter(0);
         thrust::for_each(thrust::make_zip_iterator(head, keys, iter),
                          thrust::make_zip_iterator(head+n, keys+n, iter+n),
                          thrust::make_zip_function(calc_first_pts{}));

         // Calculate distances from segment lines.
         auto hull_count_citer = thrust::make_constant_iterator<int>(hull_count);
         thrust::transform(thrust::make_zip_iterator(x, y, keys, hull_count_citer),
                           thrust::make_zip_iterator(x+n, y+n, keys+n, hull_count_citer),
                           dist,
                           thrust::make_zip_function(calc_line_dist{}));

         // Find farthest points in segments.
         thrust::device_ptr<int> reduction_border =
            thrust::reduce_by_key(// reduction keys
                                  keys, keys+n,
                                  // values input
                                  thrust::make_zip_iterator(dist, thrust::make_counting_iterator(0)),
                                  // keys output - throw away
                                  diter,
                                  // values output - only care about index
                                  thrust::make_zip_iterator(diter, flag),
                                  // use maximum to reduce
                                  thrust::equal_to<>(), thrust::maximum<>())
            .second.get_iterator_tuple().get<1>();

         // Update heads with farthest points.
         thrust::for_each(flag, reduction_border, update_heads{});

         // Determine outerior points.
         auto outerior = thrust::device_ptr<int>(reinterpret_cast<int*>(dist.get()));
         thrust::transform(thrust::make_zip_iterator(x, y, keys, head, hull_count_citer),
                           thrust::make_zip_iterator(x+n, y+n, keys+n, head+n, hull_count_citer),
                           outerior,
                           thrust::make_zip_function(calc_outerior{}));

         // Discard interior points.
         n = static_cast<int>(
               thrust::stable_partition(thrust::make_zip_iterator(x, y, head),
                                        thrust::make_zip_iterator(x+n, y+n, head+n),
                                        outerior,
                                        // move outerior points to the beginning
                                        thrust::placeholders::_1 == 1)
               .get_iterator_tuple().get<0>() - x);
      }

      // Filter potentially at most one point that is one the line between
      // its neightbours.
      if (n > 2)
      {
         auto count_iter = thrust::make_counting_iterator(0);
         auto const_iter = thrust::make_constant_iterator(n);
         hull_count = static_cast<int>(
            thrust::stable_partition(thrust::make_zip_iterator(count_iter, const_iter),
                                     thrust::make_zip_iterator(count_iter+n, const_iter+n),
                                     thrust::make_zip_function(is_on_hull{}))
            .get_iterator_tuple().get<0>() - count_iter);
      }

      timer.stop();

      if (!mem.is_host_mem)
      {
         cudaCall(cudaGraphicsUnmapResources(1, &mem.resource));
      }
      else
      {
         // Copy: CUDA->host->OpenGL.
         size_t bytes = 2 * N * sizeof(float);
         cudaCall(cudaMemcpy(mem.h_buffer, mem.d_buffer, bytes, cudaMemcpyDeviceToHost));
         glCall(glBufferSubData(GL_ARRAY_BUFFER, 0, bytes, mem.h_buffer));
      }

      glCall(glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0,
                                   reinterpret_cast<const void*>(N * sizeof(float))));

      return hull_count;
   }

   void cleanup()
   {
      if (!mem.is_host_mem)
      {
         cudaCall(cudaGraphicsUnregisterResource(mem.resource));
      }
      else
      {
         cudaCall(cudaFree(mem.d_buffer));
         cudaCall(cudaFreeHost(mem.h_buffer));
      }
   }

   __device__
   bool is_above_line::operator()(float x, float y) const
   {
      if (x == d_right_x && y == d_right_y)
         return true;
      // Unfortunately it might happen that even though (x, y) is
      // leftmost point, still cross(...) > 0 (precision problems?).
      if (x == d_left_x && y == d_left_y)
         return false;
      float ux = x - d_right_x, uy = y - d_right_y;
      float vx = d_left_x - d_right_x, vy = d_left_y - d_right_y;
      return cross(ux, uy, vx, vy) > 0;
   }

   __device__
   void calc_first_pts::operator()(int head, int key, int index) const
   {
      if (head == 1)
         d_first_pts[key] = index;
   }

   __device__
   float calc_line_dist::operator()(float x, float y, int key, int hull_count)
      const
   {
      int nxt = key + 1;
      if (nxt == hull_count) nxt = 0;

      int i = d_first_pts[key];
      int j = d_first_pts[nxt];

      // Due to precision problems we have to explicitly ensure that
      // segmenent borders will not be selected as the farthest points in the
      // segment (points stricly inside segment might have distance 0 from
      // the segment line even though they are not on it) becuase it leads
      // to an infinite loop for the main algorithm.
      float x1 = d_x[i], y1 = d_y[i];
      if (x == x1 && y == y1)
         return -1.f;
      float x2 = d_x[j], y2 = d_y[j];
      if (x == x2 && y == y2)
         return -1.f;

      float dx = x2 - x1, dy = y2 - y1;
      float ux = x1 - x, uy = y1 - y;

      return cross(dx, dy, ux, uy);
   }

   __device__
   void update_heads::operator()(int index) const
   {
      d_head[index] = 1;
   }

   __device__
   bool calc_outerior::operator()(float x, float y, int key, int head,
      int hull_count) const
   {
      if (head) return true;

      int nxt = key + 1;
      if (nxt == hull_count) nxt = 0;

      int a = d_first_pts[key];
      int b = d_first_pts[nxt];
      int c = d_flag[key];

      float cx = d_x[c], cy = d_y[c];
      float ux = d_x[a] - cx, uy = d_y[a] - cy;
      x -= cx; y -= cy;
      if (cross(ux, uy, x, y) > 0)
         return true;

      float vx = d_x[b] - cx, vy = d_y[b] - cy;
      return cross(x, y, vx, vy) > 0;
   }

   __device__
   bool is_on_hull::operator()(int index, int hull_count) const
   {
      int prv = index - 1;
      if (prv == -1) prv = hull_count - 1;
      int nxt = index + 1;
      if (nxt == hull_count) nxt = 0;

      float px = d_x[prv], py = d_y[prv];
      float ux = d_x[index] - px, uy = d_y[index] - py;
      float vx = d_x[nxt] - px, vy = d_y[nxt] - py;

      return cross(ux, uy, vx, vy) != 0;
   }
   
   __global__
   void generatePoints(float* x, float* y, const int N)
   {
      int idx = threadIdx.x + blockIdx.x * NPERTHREAD * blockDim.x;
      curandState state;
      curand_init(d_seed, idx, d_offset, &state);
      for (int i = 0; i < NPERTHREAD && idx < N; ++i)
      {
         float r = curand_uniform(&state) * d_r_diff + d_r_min;
         float a = curand_uniform(&state) * 2 * PI;
         x[idx] = r * cos(a);
         y[idx] = r * sin(a);
         idx += blockDim.x;
      }
   }

   __device__
   float cross(float ux, float uy, float vx, float vy)
   {
      return ux * vy - vx * uy;
   }

   size_t getCudaMemoryNeeded(int n)
   {
      return n * (3 * sizeof(float) + 4 * sizeof(int));
   }

} // namespace GPU
