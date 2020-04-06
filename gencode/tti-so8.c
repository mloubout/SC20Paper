#define _POSIX_C_SOURCE 200809L
#include "stdlib.h"
#include "math.h"
#include "sys/time.h"
#include "xmmintrin.h"
#include "pmmintrin.h"
#include "omp.h"

struct dataobj
{
  void *restrict data;
  int * size;
  int * npsize;
  int * dsize;
  int * hsize;
  int * hofs;
  int * oofs;
} ;

struct profiler
{
  double section0;
  double section1;
  double section2;
  double section3;
} ;

void bf0(struct dataobj *restrict damp_vec, const float dt, struct dataobj *restrict epsilon_vec, float *restrict r49_vec, float *restrict r50_vec, float *restrict r51_vec, float *restrict r52_vec, float *restrict r53_vec, struct dataobj *restrict u_vec, struct dataobj *restrict v_vec, struct dataobj *restrict vp_vec, const int x0_blk0_size, const int x_size, const int y0_blk0_size, const int y_size, const int z_size, const int t0, const int t1, const int t2, const int x_M, const int x_m, const int y_M, const int y_m, const int z_M, const int z_m, const int nthreads);

int ForwardTTI(struct dataobj *restrict damp_vec, struct dataobj *restrict delta_vec, const float dt, struct dataobj *restrict epsilon_vec, const float o_x, const float o_y, const float o_z, struct dataobj *restrict phi_vec, struct dataobj *restrict rec_vec, struct dataobj *restrict rec_coords_vec, struct dataobj *restrict src_vec, struct dataobj *restrict src_coords_vec, struct dataobj *restrict theta_vec, struct dataobj *restrict u_vec, struct dataobj *restrict v_vec, struct dataobj *restrict vp_vec, const int x0_blk0_size, const int x_M, const int x_m, const int x_size, const int y0_blk0_size, const int y_M, const int y_m, const int y_size, const int z_M, const int z_m, const int z_size, const int p_rec_M, const int p_rec_m, const int p_src_M, const int p_src_m, const int time_M, const int time_m, struct profiler * timers, const int nthreads, const int nthreads_nonaffine)
{
  float (*restrict damp)[damp_vec->size[1]][damp_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[damp_vec->size[1]][damp_vec->size[2]]) damp_vec->data;
  float (*restrict delta)[delta_vec->size[1]][delta_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[delta_vec->size[1]][delta_vec->size[2]]) delta_vec->data;
  float (*restrict epsilon)[epsilon_vec->size[1]][epsilon_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[epsilon_vec->size[1]][epsilon_vec->size[2]]) epsilon_vec->data;
  float (*restrict phi)[phi_vec->size[1]][phi_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[phi_vec->size[1]][phi_vec->size[2]]) phi_vec->data;
  float (*restrict rec)[rec_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[rec_vec->size[1]]) rec_vec->data;
  float (*restrict rec_coords)[rec_coords_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[rec_coords_vec->size[1]]) rec_coords_vec->data;
  float (*restrict src)[src_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[src_vec->size[1]]) src_vec->data;
  float (*restrict src_coords)[src_coords_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[src_coords_vec->size[1]]) src_coords_vec->data;
  float (*restrict theta)[theta_vec->size[1]][theta_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[theta_vec->size[1]][theta_vec->size[2]]) theta_vec->data;
  float (*restrict u)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]]) u_vec->data;
  float (*restrict v)[v_vec->size[1]][v_vec->size[2]][v_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[v_vec->size[1]][v_vec->size[2]][v_vec->size[3]]) v_vec->data;
  float (*restrict vp)[vp_vec->size[1]][vp_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[vp_vec->size[1]][vp_vec->size[2]]) vp_vec->data;
  float (*r53)[y_size + 2 + 2][z_size + 2 + 2];
  posix_memalign((void**)&r53, 64, sizeof(float[x_size + 2 + 2][y_size + 2 + 2][z_size + 2 + 2]));
  float (*r52)[y_size + 2 + 2][z_size + 2 + 2];
  posix_memalign((void**)&r52, 64, sizeof(float[x_size + 2 + 2][y_size + 2 + 2][z_size + 2 + 2]));
  float (*r51)[y_size + 2 + 2][z_size + 2 + 2];
  posix_memalign((void**)&r51, 64, sizeof(float[x_size + 2 + 2][y_size + 2 + 2][z_size + 2 + 2]));
  float (*r50)[y_size + 2 + 2][z_size + 2 + 2];
  posix_memalign((void**)&r50, 64, sizeof(float[x_size + 2 + 2][y_size + 2 + 2][z_size + 2 + 2]));
  float (*r49)[y_size + 2 + 2][z_size + 2 + 2];
  posix_memalign((void**)&r49, 64, sizeof(float[x_size + 2 + 2][y_size + 2 + 2][z_size + 2 + 2]));
  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  struct timeval start_section0, end_section0;
  gettimeofday(&start_section0, NULL);
  /* Begin section0 */
  #pragma omp parallel num_threads(nthreads)
  {
    #pragma omp for collapse(1) schedule(dynamic,1)
    for (int x = x_m - 2; x <= x_M + 2; x += 1)
    {
      for (int y = y_m - 2; y <= y_M + 2; y += 1)
      {
        #pragma omp simd aligned(delta,phi,theta:32)
        for (int z = z_m - 2; z <= z_M + 2; z += 1)
        {
          r53[x + 2][y + 2][z + 2] = sin(phi[x + 8][y + 8][z + 8]);
          r52[x + 2][y + 2][z + 2] = sin(theta[x + 8][y + 8][z + 8]);
          r51[x + 2][y + 2][z + 2] = cos(phi[x + 8][y + 8][z + 8]);
          r50[x + 2][y + 2][z + 2] = cos(theta[x + 8][y + 8][z + 8]);
          r49[x + 2][y + 2][z + 2] = sqrt(2*delta[x + 8][y + 8][z + 8] + 1);
        }
      }
    }
  }
  /* End section0 */
  gettimeofday(&end_section0, NULL);
  timers->section0 += (double)(end_section0.tv_sec-start_section0.tv_sec)+(double)(end_section0.tv_usec-start_section0.tv_usec)/1000000;
  for (int time = time_m, t0 = (time)%(3), t1 = (time + 1)%(3), t2 = (time + 2)%(3); time <= time_M; time += 1, t0 = (time)%(3), t1 = (time + 1)%(3), t2 = (time + 2)%(3))
  {
    struct timeval start_section1, end_section1;
    gettimeofday(&start_section1, NULL);
    /* Begin section1 */
    bf0(damp_vec,dt,epsilon_vec,(float *)r49,(float *)r50,(float *)r51,(float *)r52,(float *)r53,u_vec,v_vec,vp_vec,x0_blk0_size,x_size,y0_blk0_size,y_size,z_size,t0,t1,t2,x_M - (x_M - x_m + 1)%(x0_blk0_size),x_m,y_M - (y_M - y_m + 1)%(y0_blk0_size),y_m,z_M,z_m,nthreads);
    bf0(damp_vec,dt,epsilon_vec,(float *)r49,(float *)r50,(float *)r51,(float *)r52,(float *)r53,u_vec,v_vec,vp_vec,x0_blk0_size,x_size,(y_M - y_m + 1)%(y0_blk0_size),y_size,z_size,t0,t1,t2,x_M - (x_M - x_m + 1)%(x0_blk0_size),x_m,y_M,y_M - (y_M - y_m + 1)%(y0_blk0_size) + 1,z_M,z_m,nthreads);
    bf0(damp_vec,dt,epsilon_vec,(float *)r49,(float *)r50,(float *)r51,(float *)r52,(float *)r53,u_vec,v_vec,vp_vec,(x_M - x_m + 1)%(x0_blk0_size),x_size,y0_blk0_size,y_size,z_size,t0,t1,t2,x_M,x_M - (x_M - x_m + 1)%(x0_blk0_size) + 1,y_M - (y_M - y_m + 1)%(y0_blk0_size),y_m,z_M,z_m,nthreads);
    bf0(damp_vec,dt,epsilon_vec,(float *)r49,(float *)r50,(float *)r51,(float *)r52,(float *)r53,u_vec,v_vec,vp_vec,(x_M - x_m + 1)%(x0_blk0_size),x_size,(y_M - y_m + 1)%(y0_blk0_size),y_size,z_size,t0,t1,t2,x_M,x_M - (x_M - x_m + 1)%(x0_blk0_size) + 1,y_M,y_M - (y_M - y_m + 1)%(y0_blk0_size) + 1,z_M,z_m,nthreads);
    /* End section1 */
    gettimeofday(&end_section1, NULL);
    timers->section1 += (double)(end_section1.tv_sec-start_section1.tv_sec)+(double)(end_section1.tv_usec-start_section1.tv_usec)/1000000;
    struct timeval start_section2, end_section2;
    gettimeofday(&start_section2, NULL);
    /* Begin section2 */
    #pragma omp parallel num_threads(nthreads_nonaffine)
    {
      int chunk_size = (int)(fmax(1, (1.0F/3.0F)*(p_src_M - p_src_m + 1)/nthreads_nonaffine));
      #pragma omp for collapse(1) schedule(dynamic,chunk_size)
      for (int p_src = p_src_m; p_src <= p_src_M; p_src += 1)
      {
        int ii_src_0 = (int)(floor(-1.0e-1*o_x + 1.0e-1*src_coords[p_src][0]));
        int ii_src_1 = (int)(floor(-1.0e-1*o_y + 1.0e-1*src_coords[p_src][1]));
        int ii_src_2 = (int)(floor(-1.0e-1*o_z + 1.0e-1*src_coords[p_src][2]));
        int ii_src_3 = (int)(floor(-1.0e-1*o_z + 1.0e-1*src_coords[p_src][2])) + 1;
        int ii_src_4 = (int)(floor(-1.0e-1*o_y + 1.0e-1*src_coords[p_src][1])) + 1;
        int ii_src_5 = (int)(floor(-1.0e-1*o_x + 1.0e-1*src_coords[p_src][0])) + 1;
        float px = (float)(-o_x - 1.0e+1F*(int)(floor(-1.0e-1F*o_x + 1.0e-1F*src_coords[p_src][0])) + src_coords[p_src][0]);
        float py = (float)(-o_y - 1.0e+1F*(int)(floor(-1.0e-1F*o_y + 1.0e-1F*src_coords[p_src][1])) + src_coords[p_src][1]);
        float pz = (float)(-o_z - 1.0e+1F*(int)(floor(-1.0e-1F*o_z + 1.0e-1F*src_coords[p_src][2])) + src_coords[p_src][2]);
        if (ii_src_0 >= x_m - 1 && ii_src_1 >= y_m - 1 && ii_src_2 >= z_m - 1 && ii_src_0 <= x_M + 1 && ii_src_1 <= y_M + 1 && ii_src_2 <= z_M + 1)
        {
          float r54 = (dt*dt)*(vp[ii_src_0 + 8][ii_src_1 + 8][ii_src_2 + 8]*vp[ii_src_0 + 8][ii_src_1 + 8][ii_src_2 + 8])*(-1.0e-3F*px*py*pz + 1.0e-2F*px*py + 1.0e-2F*px*pz - 1.0e-1F*px + 1.0e-2F*py*pz - 1.0e-1F*py - 1.0e-1F*pz + 1)*src[time][p_src];
          #pragma omp atomic update
          u[t1][ii_src_0 + 8][ii_src_1 + 8][ii_src_2 + 8] += r54;
        }
        if (ii_src_0 >= x_m - 1 && ii_src_1 >= y_m - 1 && ii_src_3 >= z_m - 1 && ii_src_0 <= x_M + 1 && ii_src_1 <= y_M + 1 && ii_src_3 <= z_M + 1)
        {
          float r55 = (dt*dt)*(vp[ii_src_0 + 8][ii_src_1 + 8][ii_src_3 + 8]*vp[ii_src_0 + 8][ii_src_1 + 8][ii_src_3 + 8])*(1.0e-3F*px*py*pz - 1.0e-2F*px*pz - 1.0e-2F*py*pz + 1.0e-1F*pz)*src[time][p_src];
          #pragma omp atomic update
          u[t1][ii_src_0 + 8][ii_src_1 + 8][ii_src_3 + 8] += r55;
        }
        if (ii_src_0 >= x_m - 1 && ii_src_2 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_0 <= x_M + 1 && ii_src_2 <= z_M + 1 && ii_src_4 <= y_M + 1)
        {
          float r56 = (dt*dt)*(vp[ii_src_0 + 8][ii_src_4 + 8][ii_src_2 + 8]*vp[ii_src_0 + 8][ii_src_4 + 8][ii_src_2 + 8])*(1.0e-3F*px*py*pz - 1.0e-2F*px*py - 1.0e-2F*py*pz + 1.0e-1F*py)*src[time][p_src];
          #pragma omp atomic update
          u[t1][ii_src_0 + 8][ii_src_4 + 8][ii_src_2 + 8] += r56;
        }
        if (ii_src_0 >= x_m - 1 && ii_src_3 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_0 <= x_M + 1 && ii_src_3 <= z_M + 1 && ii_src_4 <= y_M + 1)
        {
          float r57 = (dt*dt)*(vp[ii_src_0 + 8][ii_src_4 + 8][ii_src_3 + 8]*vp[ii_src_0 + 8][ii_src_4 + 8][ii_src_3 + 8])*(-1.0e-3F*px*py*pz + 1.0e-2F*py*pz)*src[time][p_src];
          #pragma omp atomic update
          u[t1][ii_src_0 + 8][ii_src_4 + 8][ii_src_3 + 8] += r57;
        }
        if (ii_src_1 >= y_m - 1 && ii_src_2 >= z_m - 1 && ii_src_5 >= x_m - 1 && ii_src_1 <= y_M + 1 && ii_src_2 <= z_M + 1 && ii_src_5 <= x_M + 1)
        {
          float r58 = (dt*dt)*(vp[ii_src_5 + 8][ii_src_1 + 8][ii_src_2 + 8]*vp[ii_src_5 + 8][ii_src_1 + 8][ii_src_2 + 8])*(1.0e-3F*px*py*pz - 1.0e-2F*px*py - 1.0e-2F*px*pz + 1.0e-1F*px)*src[time][p_src];
          #pragma omp atomic update
          u[t1][ii_src_5 + 8][ii_src_1 + 8][ii_src_2 + 8] += r58;
        }
        if (ii_src_1 >= y_m - 1 && ii_src_3 >= z_m - 1 && ii_src_5 >= x_m - 1 && ii_src_1 <= y_M + 1 && ii_src_3 <= z_M + 1 && ii_src_5 <= x_M + 1)
        {
          float r59 = (dt*dt)*(vp[ii_src_5 + 8][ii_src_1 + 8][ii_src_3 + 8]*vp[ii_src_5 + 8][ii_src_1 + 8][ii_src_3 + 8])*(-1.0e-3F*px*py*pz + 1.0e-2F*px*pz)*src[time][p_src];
          #pragma omp atomic update
          u[t1][ii_src_5 + 8][ii_src_1 + 8][ii_src_3 + 8] += r59;
        }
        if (ii_src_2 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_5 >= x_m - 1 && ii_src_2 <= z_M + 1 && ii_src_4 <= y_M + 1 && ii_src_5 <= x_M + 1)
        {
          float r60 = (dt*dt)*(vp[ii_src_5 + 8][ii_src_4 + 8][ii_src_2 + 8]*vp[ii_src_5 + 8][ii_src_4 + 8][ii_src_2 + 8])*(-1.0e-3F*px*py*pz + 1.0e-2F*px*py)*src[time][p_src];
          #pragma omp atomic update
          u[t1][ii_src_5 + 8][ii_src_4 + 8][ii_src_2 + 8] += r60;
        }
        if (ii_src_3 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_5 >= x_m - 1 && ii_src_3 <= z_M + 1 && ii_src_4 <= y_M + 1 && ii_src_5 <= x_M + 1)
        {
          float r61 = 1.0e-3F*px*py*pz*(dt*dt)*(vp[ii_src_5 + 8][ii_src_4 + 8][ii_src_3 + 8]*vp[ii_src_5 + 8][ii_src_4 + 8][ii_src_3 + 8])*src[time][p_src];
          #pragma omp atomic update
          u[t1][ii_src_5 + 8][ii_src_4 + 8][ii_src_3 + 8] += r61;
        }
        ii_src_0 = (int)(floor(-1.0e-1*o_x + 1.0e-1*src_coords[p_src][0]));
        ii_src_1 = (int)(floor(-1.0e-1*o_y + 1.0e-1*src_coords[p_src][1]));
        ii_src_2 = (int)(floor(-1.0e-1*o_z + 1.0e-1*src_coords[p_src][2]));
        ii_src_3 = (int)(floor(-1.0e-1*o_z + 1.0e-1*src_coords[p_src][2])) + 1;
        ii_src_4 = (int)(floor(-1.0e-1*o_y + 1.0e-1*src_coords[p_src][1])) + 1;
        ii_src_5 = (int)(floor(-1.0e-1*o_x + 1.0e-1*src_coords[p_src][0])) + 1;
        px = (float)(-o_x - 1.0e+1F*(int)(floor(-1.0e-1F*o_x + 1.0e-1F*src_coords[p_src][0])) + src_coords[p_src][0]);
        py = (float)(-o_y - 1.0e+1F*(int)(floor(-1.0e-1F*o_y + 1.0e-1F*src_coords[p_src][1])) + src_coords[p_src][1]);
        pz = (float)(-o_z - 1.0e+1F*(int)(floor(-1.0e-1F*o_z + 1.0e-1F*src_coords[p_src][2])) + src_coords[p_src][2]);
        if (ii_src_0 >= x_m - 1 && ii_src_1 >= y_m - 1 && ii_src_2 >= z_m - 1 && ii_src_0 <= x_M + 1 && ii_src_1 <= y_M + 1 && ii_src_2 <= z_M + 1)
        {
          float r62 = (dt*dt)*(vp[ii_src_0 + 8][ii_src_1 + 8][ii_src_2 + 8]*vp[ii_src_0 + 8][ii_src_1 + 8][ii_src_2 + 8])*(-1.0e-3F*px*py*pz + 1.0e-2F*px*py + 1.0e-2F*px*pz - 1.0e-1F*px + 1.0e-2F*py*pz - 1.0e-1F*py - 1.0e-1F*pz + 1)*src[time][p_src];
          #pragma omp atomic update
          v[t1][ii_src_0 + 8][ii_src_1 + 8][ii_src_2 + 8] += r62;
        }
        if (ii_src_0 >= x_m - 1 && ii_src_1 >= y_m - 1 && ii_src_3 >= z_m - 1 && ii_src_0 <= x_M + 1 && ii_src_1 <= y_M + 1 && ii_src_3 <= z_M + 1)
        {
          float r63 = (dt*dt)*(vp[ii_src_0 + 8][ii_src_1 + 8][ii_src_3 + 8]*vp[ii_src_0 + 8][ii_src_1 + 8][ii_src_3 + 8])*(1.0e-3F*px*py*pz - 1.0e-2F*px*pz - 1.0e-2F*py*pz + 1.0e-1F*pz)*src[time][p_src];
          #pragma omp atomic update
          v[t1][ii_src_0 + 8][ii_src_1 + 8][ii_src_3 + 8] += r63;
        }
        if (ii_src_0 >= x_m - 1 && ii_src_2 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_0 <= x_M + 1 && ii_src_2 <= z_M + 1 && ii_src_4 <= y_M + 1)
        {
          float r64 = (dt*dt)*(vp[ii_src_0 + 8][ii_src_4 + 8][ii_src_2 + 8]*vp[ii_src_0 + 8][ii_src_4 + 8][ii_src_2 + 8])*(1.0e-3F*px*py*pz - 1.0e-2F*px*py - 1.0e-2F*py*pz + 1.0e-1F*py)*src[time][p_src];
          #pragma omp atomic update
          v[t1][ii_src_0 + 8][ii_src_4 + 8][ii_src_2 + 8] += r64;
        }
        if (ii_src_0 >= x_m - 1 && ii_src_3 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_0 <= x_M + 1 && ii_src_3 <= z_M + 1 && ii_src_4 <= y_M + 1)
        {
          float r65 = (dt*dt)*(vp[ii_src_0 + 8][ii_src_4 + 8][ii_src_3 + 8]*vp[ii_src_0 + 8][ii_src_4 + 8][ii_src_3 + 8])*(-1.0e-3F*px*py*pz + 1.0e-2F*py*pz)*src[time][p_src];
          #pragma omp atomic update
          v[t1][ii_src_0 + 8][ii_src_4 + 8][ii_src_3 + 8] += r65;
        }
        if (ii_src_1 >= y_m - 1 && ii_src_2 >= z_m - 1 && ii_src_5 >= x_m - 1 && ii_src_1 <= y_M + 1 && ii_src_2 <= z_M + 1 && ii_src_5 <= x_M + 1)
        {
          float r66 = (dt*dt)*(vp[ii_src_5 + 8][ii_src_1 + 8][ii_src_2 + 8]*vp[ii_src_5 + 8][ii_src_1 + 8][ii_src_2 + 8])*(1.0e-3F*px*py*pz - 1.0e-2F*px*py - 1.0e-2F*px*pz + 1.0e-1F*px)*src[time][p_src];
          #pragma omp atomic update
          v[t1][ii_src_5 + 8][ii_src_1 + 8][ii_src_2 + 8] += r66;
        }
        if (ii_src_1 >= y_m - 1 && ii_src_3 >= z_m - 1 && ii_src_5 >= x_m - 1 && ii_src_1 <= y_M + 1 && ii_src_3 <= z_M + 1 && ii_src_5 <= x_M + 1)
        {
          float r67 = (dt*dt)*(vp[ii_src_5 + 8][ii_src_1 + 8][ii_src_3 + 8]*vp[ii_src_5 + 8][ii_src_1 + 8][ii_src_3 + 8])*(-1.0e-3F*px*py*pz + 1.0e-2F*px*pz)*src[time][p_src];
          #pragma omp atomic update
          v[t1][ii_src_5 + 8][ii_src_1 + 8][ii_src_3 + 8] += r67;
        }
        if (ii_src_2 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_5 >= x_m - 1 && ii_src_2 <= z_M + 1 && ii_src_4 <= y_M + 1 && ii_src_5 <= x_M + 1)
        {
          float r68 = (dt*dt)*(vp[ii_src_5 + 8][ii_src_4 + 8][ii_src_2 + 8]*vp[ii_src_5 + 8][ii_src_4 + 8][ii_src_2 + 8])*(-1.0e-3F*px*py*pz + 1.0e-2F*px*py)*src[time][p_src];
          #pragma omp atomic update
          v[t1][ii_src_5 + 8][ii_src_4 + 8][ii_src_2 + 8] += r68;
        }
        if (ii_src_3 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_5 >= x_m - 1 && ii_src_3 <= z_M + 1 && ii_src_4 <= y_M + 1 && ii_src_5 <= x_M + 1)
        {
          float r69 = 1.0e-3F*px*py*pz*(dt*dt)*(vp[ii_src_5 + 8][ii_src_4 + 8][ii_src_3 + 8]*vp[ii_src_5 + 8][ii_src_4 + 8][ii_src_3 + 8])*src[time][p_src];
          #pragma omp atomic update
          v[t1][ii_src_5 + 8][ii_src_4 + 8][ii_src_3 + 8] += r69;
        }
      }
    }
    /* End section2 */
    gettimeofday(&end_section2, NULL);
    timers->section2 += (double)(end_section2.tv_sec-start_section2.tv_sec)+(double)(end_section2.tv_usec-start_section2.tv_usec)/1000000;
    struct timeval start_section3, end_section3;
    gettimeofday(&start_section3, NULL);
    /* Begin section3 */
    #pragma omp parallel num_threads(nthreads_nonaffine)
    {
      int chunk_size = (int)(fmax(1, (1.0F/3.0F)*(p_rec_M - p_rec_m + 1)/nthreads_nonaffine));
      #pragma omp for collapse(1) schedule(dynamic,chunk_size)
      for (int p_rec = p_rec_m; p_rec <= p_rec_M; p_rec += 1)
      {
        int ii_rec_0 = (int)(floor(-1.0e-1*o_x + 1.0e-1*rec_coords[p_rec][0]));
        int ii_rec_1 = (int)(floor(-1.0e-1*o_y + 1.0e-1*rec_coords[p_rec][1]));
        int ii_rec_2 = (int)(floor(-1.0e-1*o_z + 1.0e-1*rec_coords[p_rec][2]));
        int ii_rec_3 = (int)(floor(-1.0e-1*o_z + 1.0e-1*rec_coords[p_rec][2])) + 1;
        int ii_rec_4 = (int)(floor(-1.0e-1*o_y + 1.0e-1*rec_coords[p_rec][1])) + 1;
        int ii_rec_5 = (int)(floor(-1.0e-1*o_x + 1.0e-1*rec_coords[p_rec][0])) + 1;
        float px = (float)(-o_x - 1.0e+1F*(int)(floor(-1.0e-1F*o_x + 1.0e-1F*rec_coords[p_rec][0])) + rec_coords[p_rec][0]);
        float py = (float)(-o_y - 1.0e+1F*(int)(floor(-1.0e-1F*o_y + 1.0e-1F*rec_coords[p_rec][1])) + rec_coords[p_rec][1]);
        float pz = (float)(-o_z - 1.0e+1F*(int)(floor(-1.0e-1F*o_z + 1.0e-1F*rec_coords[p_rec][2])) + rec_coords[p_rec][2]);
        float sum = 0.0F;
        if (ii_rec_0 >= x_m - 1 && ii_rec_1 >= y_m - 1 && ii_rec_2 >= z_m - 1 && ii_rec_0 <= x_M + 1 && ii_rec_1 <= y_M + 1 && ii_rec_2 <= z_M + 1)
        {
          sum += (u[t0][ii_rec_0 + 8][ii_rec_1 + 8][ii_rec_2 + 8] + v[t0][ii_rec_0 + 8][ii_rec_1 + 8][ii_rec_2 + 8])*(-1.0e-3F*px*py*pz + 1.0e-2F*px*py + 1.0e-2F*px*pz - 1.0e-1F*px + 1.0e-2F*py*pz - 1.0e-1F*py - 1.0e-1F*pz + 1);
        }
        if (ii_rec_0 >= x_m - 1 && ii_rec_1 >= y_m - 1 && ii_rec_3 >= z_m - 1 && ii_rec_0 <= x_M + 1 && ii_rec_1 <= y_M + 1 && ii_rec_3 <= z_M + 1)
        {
          sum += (u[t0][ii_rec_0 + 8][ii_rec_1 + 8][ii_rec_3 + 8] + v[t0][ii_rec_0 + 8][ii_rec_1 + 8][ii_rec_3 + 8])*(1.0e-3F*px*py*pz - 1.0e-2F*px*pz - 1.0e-2F*py*pz + 1.0e-1F*pz);
        }
        if (ii_rec_0 >= x_m - 1 && ii_rec_2 >= z_m - 1 && ii_rec_4 >= y_m - 1 && ii_rec_0 <= x_M + 1 && ii_rec_2 <= z_M + 1 && ii_rec_4 <= y_M + 1)
        {
          sum += (u[t0][ii_rec_0 + 8][ii_rec_4 + 8][ii_rec_2 + 8] + v[t0][ii_rec_0 + 8][ii_rec_4 + 8][ii_rec_2 + 8])*(1.0e-3F*px*py*pz - 1.0e-2F*px*py - 1.0e-2F*py*pz + 1.0e-1F*py);
        }
        if (ii_rec_0 >= x_m - 1 && ii_rec_3 >= z_m - 1 && ii_rec_4 >= y_m - 1 && ii_rec_0 <= x_M + 1 && ii_rec_3 <= z_M + 1 && ii_rec_4 <= y_M + 1)
        {
          sum += (-1.0e-3F*px*py*pz + 1.0e-2F*py*pz)*(u[t0][ii_rec_0 + 8][ii_rec_4 + 8][ii_rec_3 + 8] + v[t0][ii_rec_0 + 8][ii_rec_4 + 8][ii_rec_3 + 8]);
        }
        if (ii_rec_1 >= y_m - 1 && ii_rec_2 >= z_m - 1 && ii_rec_5 >= x_m - 1 && ii_rec_1 <= y_M + 1 && ii_rec_2 <= z_M + 1 && ii_rec_5 <= x_M + 1)
        {
          sum += (u[t0][ii_rec_5 + 8][ii_rec_1 + 8][ii_rec_2 + 8] + v[t0][ii_rec_5 + 8][ii_rec_1 + 8][ii_rec_2 + 8])*(1.0e-3F*px*py*pz - 1.0e-2F*px*py - 1.0e-2F*px*pz + 1.0e-1F*px);
        }
        if (ii_rec_1 >= y_m - 1 && ii_rec_3 >= z_m - 1 && ii_rec_5 >= x_m - 1 && ii_rec_1 <= y_M + 1 && ii_rec_3 <= z_M + 1 && ii_rec_5 <= x_M + 1)
        {
          sum += (-1.0e-3F*px*py*pz + 1.0e-2F*px*pz)*(u[t0][ii_rec_5 + 8][ii_rec_1 + 8][ii_rec_3 + 8] + v[t0][ii_rec_5 + 8][ii_rec_1 + 8][ii_rec_3 + 8]);
        }
        if (ii_rec_2 >= z_m - 1 && ii_rec_4 >= y_m - 1 && ii_rec_5 >= x_m - 1 && ii_rec_2 <= z_M + 1 && ii_rec_4 <= y_M + 1 && ii_rec_5 <= x_M + 1)
        {
          sum += (-1.0e-3F*px*py*pz + 1.0e-2F*px*py)*(u[t0][ii_rec_5 + 8][ii_rec_4 + 8][ii_rec_2 + 8] + v[t0][ii_rec_5 + 8][ii_rec_4 + 8][ii_rec_2 + 8]);
        }
        if (ii_rec_3 >= z_m - 1 && ii_rec_4 >= y_m - 1 && ii_rec_5 >= x_m - 1 && ii_rec_3 <= z_M + 1 && ii_rec_4 <= y_M + 1 && ii_rec_5 <= x_M + 1)
        {
          sum += 1.0e-3F*px*py*pz*(u[t0][ii_rec_5 + 8][ii_rec_4 + 8][ii_rec_3 + 8] + v[t0][ii_rec_5 + 8][ii_rec_4 + 8][ii_rec_3 + 8]);
        }
        rec[time][p_rec] = sum;
      }
    }
    /* End section3 */
    gettimeofday(&end_section3, NULL);
    timers->section3 += (double)(end_section3.tv_sec-start_section3.tv_sec)+(double)(end_section3.tv_usec-start_section3.tv_usec)/1000000;
  }
  free(r53);
  free(r52);
  free(r51);
  free(r50);
  free(r49);
  return 0;
}

void bf0(struct dataobj *restrict damp_vec, const float dt, struct dataobj *restrict epsilon_vec, float *restrict r49_vec, float *restrict r50_vec, float *restrict r51_vec, float *restrict r52_vec, float *restrict r53_vec, struct dataobj *restrict u_vec, struct dataobj *restrict v_vec, struct dataobj *restrict vp_vec, const int x0_blk0_size, const int x_size, const int y0_blk0_size, const int y_size, const int z_size, const int t0, const int t1, const int t2, const int x_M, const int x_m, const int y_M, const int y_m, const int z_M, const int z_m, const int nthreads)
{
  float (*restrict damp)[damp_vec->size[1]][damp_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[damp_vec->size[1]][damp_vec->size[2]]) damp_vec->data;
  float (*restrict epsilon)[epsilon_vec->size[1]][epsilon_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[epsilon_vec->size[1]][epsilon_vec->size[2]]) epsilon_vec->data;
  float (*restrict r49)[y_size + 2 + 2][z_size + 2 + 2] __attribute__ ((aligned (64))) = (float (*)[y_size + 2 + 2][z_size + 2 + 2]) r49_vec;
  float (*restrict r50)[y_size + 2 + 2][z_size + 2 + 2] __attribute__ ((aligned (64))) = (float (*)[y_size + 2 + 2][z_size + 2 + 2]) r50_vec;
  float (*restrict r51)[y_size + 2 + 2][z_size + 2 + 2] __attribute__ ((aligned (64))) = (float (*)[y_size + 2 + 2][z_size + 2 + 2]) r51_vec;
  float (*restrict r52)[y_size + 2 + 2][z_size + 2 + 2] __attribute__ ((aligned (64))) = (float (*)[y_size + 2 + 2][z_size + 2 + 2]) r52_vec;
  float (*restrict r53)[y_size + 2 + 2][z_size + 2 + 2] __attribute__ ((aligned (64))) = (float (*)[y_size + 2 + 2][z_size + 2 + 2]) r53_vec;
  float (*restrict u)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]]) u_vec->data;
  float (*restrict v)[v_vec->size[1]][v_vec->size[2]][v_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[v_vec->size[1]][v_vec->size[2]][v_vec->size[3]]) v_vec->data;
  float (*restrict vp)[vp_vec->size[1]][vp_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[vp_vec->size[1]][vp_vec->size[2]]) vp_vec->data;
  float r179[x0_blk0_size + 2 + 2][y0_blk0_size + 2 + 2][z_size + 2 + 2] __attribute__((aligned(64)));
  float r178[x0_blk0_size + 2 + 2][y0_blk0_size + 2 + 2][z_size + 2 + 2] __attribute__((aligned(64)));
  if (x0_blk0_size == 0)
  {
    return;
  }
  #pragma omp parallel num_threads(nthreads) private(r178,r179)
  {
    #pragma omp for collapse(1) schedule(dynamic,1)
    for (int x0_blk0 = x_m; x0_blk0 <= x_M; x0_blk0 += x0_blk0_size)
    {
      for (int y0_blk0 = y_m; y0_blk0 <= y_M; y0_blk0 += y0_blk0_size)
      {
        for (int x = x0_blk0 - 2, xs = 0; x <= x0_blk0 + x0_blk0_size + 1; x += 1, xs += 1)
        {
          for (int y = y0_blk0 - 2, ys = 0; y <= y0_blk0 + y0_blk0_size + 1; y += 1, ys += 1)
          {
            #pragma omp simd aligned(u,v:32)
            for (int z = z_m - 2; z <= z_M + 2; z += 1)
            {
              float r187 = 8.33333346e-3F*(v[t0][x + 10][y + 8][z + 8] - v[t0][x + 10][y + 12][z + 8]) + 6.66666677e-2F*(-v[t0][x + 10][y + 9][z + 8] + v[t0][x + 10][y + 11][z + 8]);
              float r188 = 8.33333346e-3F*(v[t0][x + 8][y + 10][z + 8] - v[t0][x + 12][y + 10][z + 8]) + 6.66666677e-2F*(-v[t0][x + 9][y + 10][z + 8] + v[t0][x + 11][y + 10][z + 8]);
              float r189 = 8.33333346e-3F*(v[t0][x + 10][y + 10][z + 6] - v[t0][x + 10][y + 10][z + 10]) + 6.66666677e-2F*(-v[t0][x + 10][y + 10][z + 7] + v[t0][x + 10][y + 10][z + 9]);
              float r190 = 8.33333346e-3F*(u[t0][x + 10][y + 8][z + 8] - u[t0][x + 10][y + 12][z + 8]) + 6.66666677e-2F*(-u[t0][x + 10][y + 9][z + 8] + u[t0][x + 10][y + 11][z + 8]);
              float r191 = 8.33333346e-3F*(u[t0][x + 8][y + 10][z + 8] - u[t0][x + 12][y + 10][z + 8]) + 6.66666677e-2F*(-u[t0][x + 9][y + 10][z + 8] + u[t0][x + 11][y + 10][z + 8]);
              float r192 = 8.33333346e-3F*(u[t0][x + 10][y + 10][z + 6] - u[t0][x + 10][y + 10][z + 10]) + 6.66666677e-2F*(-u[t0][x + 10][y + 10][z + 7] + u[t0][x + 10][y + 10][z + 9]);
              r179[xs][ys][z + 2] = -(r187*r52[x + 2][y + 2][z + 2]*r53[x + 2][y + 2][z + 2] + r188*r51[x + 2][y + 2][z + 2]*r52[x + 2][y + 2][z + 2] + r189*r50[x + 2][y + 2][z + 2]);
              r178[xs][ys][z + 2] = -(r190*r52[x + 2][y + 2][z + 2]*r53[x + 2][y + 2][z + 2] + r191*r51[x + 2][y + 2][z + 2]*r52[x + 2][y + 2][z + 2] + r192*r50[x + 2][y + 2][z + 2]);
            }
          }
        }
        for (int x = x0_blk0, xs = 0; x <= x0_blk0 + x0_blk0_size - 1; x += 1, xs += 1)
        {
          for (int y = y0_blk0, ys = 0; y <= y0_blk0 + y0_blk0_size - 1; y += 1, ys += 1)
          {
            #pragma omp simd aligned(damp,epsilon,u,v,vp:32)
            for (int z = z_m; z <= z_M; z += 1)
            {
              float r186 = dt*dt;
              float r185 = dt*damp[x + 1][y + 1][z + 1];
              float r184 = r179[xs + 1][ys + 2][z + 2]*r51[x + 1][y + 2][z + 2]*r52[x + 1][y + 2][z + 2] + r179[xs + 2][ys + 1][z + 2]*r52[x + 2][y + 1][z + 2]*r53[x + 2][y + 1][z + 2] + r179[xs + 2][ys + 2][z + 1]*r50[x + 2][y + 2][z + 1] - r179[xs + 2][ys + 2][z + 3]*r50[x + 2][y + 2][z + 3] - r179[xs + 2][ys + 3][z + 2]*r52[x + 2][y + 3][z + 2]*r53[x + 2][y + 3][z + 2] - r179[xs + 3][ys + 2][z + 2]*r51[x + 3][y + 2][z + 2]*r52[x + 3][y + 2][z + 2];
              float r183 = -r179[xs][ys + 2][z + 2]*r51[x][y + 2][z + 2]*r52[x][y + 2][z + 2] - r179[xs + 2][ys][z + 2]*r52[x + 2][y][z + 2]*r53[x + 2][y][z + 2] - r179[xs + 2][ys + 2][z]*r50[x + 2][y + 2][z] + r179[xs + 2][ys + 2][z + 4]*r50[x + 2][y + 2][z + 4] + r179[xs + 2][ys + 4][z + 2]*r52[x + 2][y + 4][z + 2]*r53[x + 2][y + 4][z + 2] + r179[xs + 4][ys + 2][z + 2]*r51[x + 4][y + 2][z + 2]*r52[x + 4][y + 2][z + 2];
              float r182 = 6.66666677e-2F*(-r178[xs + 1][ys + 2][z + 2]*r51[x + 1][y + 2][z + 2]*r52[x + 1][y + 2][z + 2] - r178[xs + 2][ys + 1][z + 2]*r52[x + 2][y + 1][z + 2]*r53[x + 2][y + 1][z + 2] - r178[xs + 2][ys + 2][z + 1]*r50[x + 2][y + 2][z + 1] + r178[xs + 2][ys + 2][z + 3]*r50[x + 2][y + 2][z + 3] + r178[xs + 2][ys + 3][z + 2]*r52[x + 2][y + 3][z + 2]*r53[x + 2][y + 3][z + 2] + r178[xs + 3][ys + 2][z + 2]*r51[x + 3][y + 2][z + 2]*r52[x + 3][y + 2][z + 2]);
              float r181 = 8.33333346e-3F*(r178[xs][ys + 2][z + 2]*r51[x][y + 2][z + 2]*r52[x][y + 2][z + 2] + r178[xs + 2][ys][z + 2]*r52[x + 2][y][z + 2]*r53[x + 2][y][z + 2] + r178[xs + 2][ys + 2][z]*r50[x + 2][y + 2][z] - r178[xs + 2][ys + 2][z + 4]*r50[x + 2][y + 2][z + 4] - r178[xs + 2][ys + 4][z + 2]*r52[x + 2][y + 4][z + 2]*r53[x + 2][y + 4][z + 2] - r178[xs + 4][ys + 2][z + 2]*r51[x + 4][y + 2][z + 2]*r52[x + 4][y + 2][z + 2]);
              float r180 = 1.0/(vp[x + 8][y + 8][z + 8]*vp[x + 8][y + 8][z + 8]);
              float r151 = 2.0F*r180 + r185;
              float r152 = -2.0F*r180 + r185;
              float r153 = -1.78571425e-5F*(u[t0][x + 4][y + 8][z + 8] + u[t0][x + 8][y + 4][z + 8] + u[t0][x + 8][y + 8][z + 4] + u[t0][x + 8][y + 8][z + 12] + u[t0][x + 8][y + 12][z + 8] + u[t0][x + 12][y + 8][z + 8]) + 2.53968248e-4F*(u[t0][x + 5][y + 8][z + 8] + u[t0][x + 8][y + 5][z + 8] + u[t0][x + 8][y + 8][z + 5] + u[t0][x + 8][y + 8][z + 11] + u[t0][x + 8][y + 11][z + 8] + u[t0][x + 11][y + 8][z + 8]) - 1.99999996e-3F*(u[t0][x + 6][y + 8][z + 8] + u[t0][x + 8][y + 6][z + 8] + u[t0][x + 8][y + 8][z + 6] + u[t0][x + 8][y + 8][z + 10] + u[t0][x + 8][y + 10][z + 8] + u[t0][x + 10][y + 8][z + 8]) + 1.59999996e-2F*(u[t0][x + 7][y + 8][z + 8] + u[t0][x + 8][y + 7][z + 8] + u[t0][x + 8][y + 8][z + 7] + u[t0][x + 8][y + 8][z + 9] + u[t0][x + 8][y + 9][z + 8] + u[t0][x + 9][y + 8][z + 8]) - 8.54166647e-2F*u[t0][x + 8][y + 8][z + 8];
              u[t1][x + 8][y + 8][z + 8] = 1.0F*(r152*u[t2][x + 8][y + 8][z + 8] + 4.0F*r180*u[t0][x + 8][y + 8][z + 8] + 2.0F*r186*((8.33333346e-3F*r183 + 6.66666677e-2F*r184)*r49[x + 2][y + 2][z + 2] + (2*epsilon[x + 8][y + 8][z + 8] + 1)*(r153 + r181 + r182)))/r151;
              v[t1][x + 8][y + 8][z + 8] = 1.0F*(r152*v[t2][x + 8][y + 8][z + 8] + 4.0F*r180*v[t0][x + 8][y + 8][z + 8] + r186*(1.66666669138067e-2F*r183 + 1.33333335310454e-1F*r184 + 2.0F*(r153 + r181 + r182)*r49[x + 2][y + 2][z + 2]))/r151;
            }
          }
        }
      }
    }
  }
}
