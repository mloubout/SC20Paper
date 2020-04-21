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

void bf0(struct dataobj *restrict damp_vec, struct dataobj *restrict irho_vec, struct dataobj *restrict tau_xx_vec, struct dataobj *restrict tau_xy_vec, struct dataobj *restrict tau_xz_vec, struct dataobj *restrict tau_yy_vec, struct dataobj *restrict tau_yz_vec, struct dataobj *restrict tau_zz_vec, struct dataobj *restrict v_x_vec, struct dataobj *restrict v_y_vec, struct dataobj *restrict v_z_vec, const int t0, const int t1, const int x0_blk0_size, const int x_M, const int x_m, const int y0_blk0_size, const int y_M, const int y_m, const int z_M, const int z_m, const int nthreads);
void bf1(struct dataobj *restrict damp_vec, struct dataobj *restrict lam_vec, struct dataobj *restrict mu_vec, struct dataobj *restrict tau_xx_vec, struct dataobj *restrict tau_xy_vec, struct dataobj *restrict tau_xz_vec, struct dataobj *restrict tau_yy_vec, struct dataobj *restrict tau_yz_vec, struct dataobj *restrict tau_zz_vec, struct dataobj *restrict v_x_vec, struct dataobj *restrict v_y_vec, struct dataobj *restrict v_z_vec, const int t0, const int t1, const int x1_blk0_size, const int x_M, const int x_m, const int y1_blk0_size, const int y_M, const int y_m, const int z_M, const int z_m, const int nthreads);

int ForwardElastic(struct dataobj *restrict damp_vec, const float dt, struct dataobj *restrict irho_vec, struct dataobj *restrict lam_vec, struct dataobj *restrict mu_vec, const float o_x, const float o_y, const float o_z, struct dataobj *restrict rec1_vec, struct dataobj *restrict rec1_coords_vec, struct dataobj *restrict rec2_vec, struct dataobj *restrict rec2_coords_vec, struct dataobj *restrict src_vec, struct dataobj *restrict src_coords_vec, struct dataobj *restrict tau_xx_vec, struct dataobj *restrict tau_xy_vec, struct dataobj *restrict tau_xz_vec, struct dataobj *restrict tau_yy_vec, struct dataobj *restrict tau_yz_vec, struct dataobj *restrict tau_zz_vec, struct dataobj *restrict v_x_vec, struct dataobj *restrict v_y_vec, struct dataobj *restrict v_z_vec, const int x_M, const int x_m, const int y_M, const int y_m, const int z_M, const int z_m, const int p_rec1_M, const int p_rec1_m, const int p_rec2_M, const int p_rec2_m, const int p_src_M, const int p_src_m, const int time_M, const int time_m, struct profiler * timers, const int x0_blk0_size, const int x1_blk0_size, const int y0_blk0_size, const int y1_blk0_size, const int nthreads, const int nthreads_nonaffine)
{
  float (*restrict rec1)[rec1_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[rec1_vec->size[1]]) rec1_vec->data;
  float (*restrict rec1_coords)[rec1_coords_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[rec1_coords_vec->size[1]]) rec1_coords_vec->data;
  float (*restrict rec2)[rec2_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[rec2_vec->size[1]]) rec2_vec->data;
  float (*restrict rec2_coords)[rec2_coords_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[rec2_coords_vec->size[1]]) rec2_coords_vec->data;
  float (*restrict src)[src_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[src_vec->size[1]]) src_vec->data;
  float (*restrict src_coords)[src_coords_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[src_coords_vec->size[1]]) src_coords_vec->data;
  float (*restrict tau_xx)[tau_xx_vec->size[1]][tau_xx_vec->size[2]][tau_xx_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[tau_xx_vec->size[1]][tau_xx_vec->size[2]][tau_xx_vec->size[3]]) tau_xx_vec->data;
  float (*restrict tau_yy)[tau_yy_vec->size[1]][tau_yy_vec->size[2]][tau_yy_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[tau_yy_vec->size[1]][tau_yy_vec->size[2]][tau_yy_vec->size[3]]) tau_yy_vec->data;
  float (*restrict tau_zz)[tau_zz_vec->size[1]][tau_zz_vec->size[2]][tau_zz_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[tau_zz_vec->size[1]][tau_zz_vec->size[2]][tau_zz_vec->size[3]]) tau_zz_vec->data;
  float (*restrict v_x)[v_x_vec->size[1]][v_x_vec->size[2]][v_x_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[v_x_vec->size[1]][v_x_vec->size[2]][v_x_vec->size[3]]) v_x_vec->data;
  float (*restrict v_y)[v_y_vec->size[1]][v_y_vec->size[2]][v_y_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[v_y_vec->size[1]][v_y_vec->size[2]][v_y_vec->size[3]]) v_y_vec->data;
  float (*restrict v_z)[v_z_vec->size[1]][v_z_vec->size[2]][v_z_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[v_z_vec->size[1]][v_z_vec->size[2]][v_z_vec->size[3]]) v_z_vec->data;
  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  for (int time = time_m, t0 = (time)%(2), t1 = (time + 1)%(2); time <= time_M; time += 1, t0 = (time)%(2), t1 = (time + 1)%(2))
  {
    struct timeval start_section0, end_section0;
    gettimeofday(&start_section0, NULL);
    /* Begin section0 */
    bf0(damp_vec,irho_vec,tau_xx_vec,tau_xy_vec,tau_xz_vec,tau_yy_vec,tau_yz_vec,tau_zz_vec,v_x_vec,v_y_vec,v_z_vec,t0,t1,x0_blk0_size,x_M - (x_M - x_m + 1)%(x0_blk0_size),x_m,y0_blk0_size,y_M - (y_M - y_m + 1)%(y0_blk0_size),y_m,z_M,z_m,nthreads);
    bf0(damp_vec,irho_vec,tau_xx_vec,tau_xy_vec,tau_xz_vec,tau_yy_vec,tau_yz_vec,tau_zz_vec,v_x_vec,v_y_vec,v_z_vec,t0,t1,x0_blk0_size,x_M - (x_M - x_m + 1)%(x0_blk0_size),x_m,(y_M - y_m + 1)%(y0_blk0_size),y_M,y_M - (y_M - y_m + 1)%(y0_blk0_size) + 1,z_M,z_m,nthreads);
    bf0(damp_vec,irho_vec,tau_xx_vec,tau_xy_vec,tau_xz_vec,tau_yy_vec,tau_yz_vec,tau_zz_vec,v_x_vec,v_y_vec,v_z_vec,t0,t1,(x_M - x_m + 1)%(x0_blk0_size),x_M,x_M - (x_M - x_m + 1)%(x0_blk0_size) + 1,y0_blk0_size,y_M - (y_M - y_m + 1)%(y0_blk0_size),y_m,z_M,z_m,nthreads);
    bf0(damp_vec,irho_vec,tau_xx_vec,tau_xy_vec,tau_xz_vec,tau_yy_vec,tau_yz_vec,tau_zz_vec,v_x_vec,v_y_vec,v_z_vec,t0,t1,(x_M - x_m + 1)%(x0_blk0_size),x_M,x_M - (x_M - x_m + 1)%(x0_blk0_size) + 1,(y_M - y_m + 1)%(y0_blk0_size),y_M,y_M - (y_M - y_m + 1)%(y0_blk0_size) + 1,z_M,z_m,nthreads);
    bf1(damp_vec,lam_vec,mu_vec,tau_xx_vec,tau_xy_vec,tau_xz_vec,tau_yy_vec,tau_yz_vec,tau_zz_vec,v_x_vec,v_y_vec,v_z_vec,t0,t1,x1_blk0_size,x_M - (x_M - x_m + 1)%(x1_blk0_size),x_m,y1_blk0_size,y_M - (y_M - y_m + 1)%(y1_blk0_size),y_m,z_M,z_m,nthreads);
    bf1(damp_vec,lam_vec,mu_vec,tau_xx_vec,tau_xy_vec,tau_xz_vec,tau_yy_vec,tau_yz_vec,tau_zz_vec,v_x_vec,v_y_vec,v_z_vec,t0,t1,x1_blk0_size,x_M - (x_M - x_m + 1)%(x1_blk0_size),x_m,(y_M - y_m + 1)%(y1_blk0_size),y_M,y_M - (y_M - y_m + 1)%(y1_blk0_size) + 1,z_M,z_m,nthreads);
    bf1(damp_vec,lam_vec,mu_vec,tau_xx_vec,tau_xy_vec,tau_xz_vec,tau_yy_vec,tau_yz_vec,tau_zz_vec,v_x_vec,v_y_vec,v_z_vec,t0,t1,(x_M - x_m + 1)%(x1_blk0_size),x_M,x_M - (x_M - x_m + 1)%(x1_blk0_size) + 1,y1_blk0_size,y_M - (y_M - y_m + 1)%(y1_blk0_size),y_m,z_M,z_m,nthreads);
    bf1(damp_vec,lam_vec,mu_vec,tau_xx_vec,tau_xy_vec,tau_xz_vec,tau_yy_vec,tau_yz_vec,tau_zz_vec,v_x_vec,v_y_vec,v_z_vec,t0,t1,(x_M - x_m + 1)%(x1_blk0_size),x_M,x_M - (x_M - x_m + 1)%(x1_blk0_size) + 1,(y_M - y_m + 1)%(y1_blk0_size),y_M,y_M - (y_M - y_m + 1)%(y1_blk0_size) + 1,z_M,z_m,nthreads);
    /* End section0 */
    gettimeofday(&end_section0, NULL);
    timers->section0 += (double)(end_section0.tv_sec-start_section0.tv_sec)+(double)(end_section0.tv_usec-start_section0.tv_usec)/1000000;
    struct timeval start_section1, end_section1;
    gettimeofday(&start_section1, NULL);
    /* Begin section1 */
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
          float r0 = dt*(-1.0e-3F*px*py*pz + 1.0e-2F*px*py + 1.0e-2F*px*pz - 1.0e-1F*px + 1.0e-2F*py*pz - 1.0e-1F*py - 1.0e-1F*pz + 1)*src[time][p_src];
          #pragma omp atomic update
          tau_xx[t1][ii_src_0 + 12][ii_src_1 + 12][ii_src_2 + 12] += r0;
        }
        if (ii_src_0 >= x_m - 1 && ii_src_1 >= y_m - 1 && ii_src_3 >= z_m - 1 && ii_src_0 <= x_M + 1 && ii_src_1 <= y_M + 1 && ii_src_3 <= z_M + 1)
        {
          float r1 = dt*(1.0e-3F*px*py*pz - 1.0e-2F*px*pz - 1.0e-2F*py*pz + 1.0e-1F*pz)*src[time][p_src];
          #pragma omp atomic update
          tau_xx[t1][ii_src_0 + 12][ii_src_1 + 12][ii_src_3 + 12] += r1;
        }
        if (ii_src_0 >= x_m - 1 && ii_src_2 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_0 <= x_M + 1 && ii_src_2 <= z_M + 1 && ii_src_4 <= y_M + 1)
        {
          float r2 = dt*(1.0e-3F*px*py*pz - 1.0e-2F*px*py - 1.0e-2F*py*pz + 1.0e-1F*py)*src[time][p_src];
          #pragma omp atomic update
          tau_xx[t1][ii_src_0 + 12][ii_src_4 + 12][ii_src_2 + 12] += r2;
        }
        if (ii_src_0 >= x_m - 1 && ii_src_3 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_0 <= x_M + 1 && ii_src_3 <= z_M + 1 && ii_src_4 <= y_M + 1)
        {
          float r3 = dt*(-1.0e-3F*px*py*pz + 1.0e-2F*py*pz)*src[time][p_src];
          #pragma omp atomic update
          tau_xx[t1][ii_src_0 + 12][ii_src_4 + 12][ii_src_3 + 12] += r3;
        }
        if (ii_src_1 >= y_m - 1 && ii_src_2 >= z_m - 1 && ii_src_5 >= x_m - 1 && ii_src_1 <= y_M + 1 && ii_src_2 <= z_M + 1 && ii_src_5 <= x_M + 1)
        {
          float r4 = dt*(1.0e-3F*px*py*pz - 1.0e-2F*px*py - 1.0e-2F*px*pz + 1.0e-1F*px)*src[time][p_src];
          #pragma omp atomic update
          tau_xx[t1][ii_src_5 + 12][ii_src_1 + 12][ii_src_2 + 12] += r4;
        }
        if (ii_src_1 >= y_m - 1 && ii_src_3 >= z_m - 1 && ii_src_5 >= x_m - 1 && ii_src_1 <= y_M + 1 && ii_src_3 <= z_M + 1 && ii_src_5 <= x_M + 1)
        {
          float r5 = dt*(-1.0e-3F*px*py*pz + 1.0e-2F*px*pz)*src[time][p_src];
          #pragma omp atomic update
          tau_xx[t1][ii_src_5 + 12][ii_src_1 + 12][ii_src_3 + 12] += r5;
        }
        if (ii_src_2 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_5 >= x_m - 1 && ii_src_2 <= z_M + 1 && ii_src_4 <= y_M + 1 && ii_src_5 <= x_M + 1)
        {
          float r6 = dt*(-1.0e-3F*px*py*pz + 1.0e-2F*px*py)*src[time][p_src];
          #pragma omp atomic update
          tau_xx[t1][ii_src_5 + 12][ii_src_4 + 12][ii_src_2 + 12] += r6;
        }
        if (ii_src_3 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_5 >= x_m - 1 && ii_src_3 <= z_M + 1 && ii_src_4 <= y_M + 1 && ii_src_5 <= x_M + 1)
        {
          float r7 = 1.0e-3F*px*py*pz*dt*src[time][p_src];
          #pragma omp atomic update
          tau_xx[t1][ii_src_5 + 12][ii_src_4 + 12][ii_src_3 + 12] += r7;
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
          float r8 = dt*(-1.0e-3F*px*py*pz + 1.0e-2F*px*py + 1.0e-2F*px*pz - 1.0e-1F*px + 1.0e-2F*py*pz - 1.0e-1F*py - 1.0e-1F*pz + 1)*src[time][p_src];
          #pragma omp atomic update
          tau_zz[t1][ii_src_0 + 12][ii_src_1 + 12][ii_src_2 + 12] += r8;
        }
        if (ii_src_0 >= x_m - 1 && ii_src_1 >= y_m - 1 && ii_src_3 >= z_m - 1 && ii_src_0 <= x_M + 1 && ii_src_1 <= y_M + 1 && ii_src_3 <= z_M + 1)
        {
          float r9 = dt*(1.0e-3F*px*py*pz - 1.0e-2F*px*pz - 1.0e-2F*py*pz + 1.0e-1F*pz)*src[time][p_src];
          #pragma omp atomic update
          tau_zz[t1][ii_src_0 + 12][ii_src_1 + 12][ii_src_3 + 12] += r9;
        }
        if (ii_src_0 >= x_m - 1 && ii_src_2 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_0 <= x_M + 1 && ii_src_2 <= z_M + 1 && ii_src_4 <= y_M + 1)
        {
          float r10 = dt*(1.0e-3F*px*py*pz - 1.0e-2F*px*py - 1.0e-2F*py*pz + 1.0e-1F*py)*src[time][p_src];
          #pragma omp atomic update
          tau_zz[t1][ii_src_0 + 12][ii_src_4 + 12][ii_src_2 + 12] += r10;
        }
        if (ii_src_0 >= x_m - 1 && ii_src_3 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_0 <= x_M + 1 && ii_src_3 <= z_M + 1 && ii_src_4 <= y_M + 1)
        {
          float r11 = dt*(-1.0e-3F*px*py*pz + 1.0e-2F*py*pz)*src[time][p_src];
          #pragma omp atomic update
          tau_zz[t1][ii_src_0 + 12][ii_src_4 + 12][ii_src_3 + 12] += r11;
        }
        if (ii_src_1 >= y_m - 1 && ii_src_2 >= z_m - 1 && ii_src_5 >= x_m - 1 && ii_src_1 <= y_M + 1 && ii_src_2 <= z_M + 1 && ii_src_5 <= x_M + 1)
        {
          float r12 = dt*(1.0e-3F*px*py*pz - 1.0e-2F*px*py - 1.0e-2F*px*pz + 1.0e-1F*px)*src[time][p_src];
          #pragma omp atomic update
          tau_zz[t1][ii_src_5 + 12][ii_src_1 + 12][ii_src_2 + 12] += r12;
        }
        if (ii_src_1 >= y_m - 1 && ii_src_3 >= z_m - 1 && ii_src_5 >= x_m - 1 && ii_src_1 <= y_M + 1 && ii_src_3 <= z_M + 1 && ii_src_5 <= x_M + 1)
        {
          float r13 = dt*(-1.0e-3F*px*py*pz + 1.0e-2F*px*pz)*src[time][p_src];
          #pragma omp atomic update
          tau_zz[t1][ii_src_5 + 12][ii_src_1 + 12][ii_src_3 + 12] += r13;
        }
        if (ii_src_2 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_5 >= x_m - 1 && ii_src_2 <= z_M + 1 && ii_src_4 <= y_M + 1 && ii_src_5 <= x_M + 1)
        {
          float r14 = dt*(-1.0e-3F*px*py*pz + 1.0e-2F*px*py)*src[time][p_src];
          #pragma omp atomic update
          tau_zz[t1][ii_src_5 + 12][ii_src_4 + 12][ii_src_2 + 12] += r14;
        }
        if (ii_src_3 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_5 >= x_m - 1 && ii_src_3 <= z_M + 1 && ii_src_4 <= y_M + 1 && ii_src_5 <= x_M + 1)
        {
          float r15 = 1.0e-3F*px*py*pz*dt*src[time][p_src];
          #pragma omp atomic update
          tau_zz[t1][ii_src_5 + 12][ii_src_4 + 12][ii_src_3 + 12] += r15;
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
          float r16 = dt*(-1.0e-3F*px*py*pz + 1.0e-2F*px*py + 1.0e-2F*px*pz - 1.0e-1F*px + 1.0e-2F*py*pz - 1.0e-1F*py - 1.0e-1F*pz + 1)*src[time][p_src];
          #pragma omp atomic update
          tau_yy[t1][ii_src_0 + 12][ii_src_1 + 12][ii_src_2 + 12] += r16;
        }
        if (ii_src_0 >= x_m - 1 && ii_src_1 >= y_m - 1 && ii_src_3 >= z_m - 1 && ii_src_0 <= x_M + 1 && ii_src_1 <= y_M + 1 && ii_src_3 <= z_M + 1)
        {
          float r17 = dt*(1.0e-3F*px*py*pz - 1.0e-2F*px*pz - 1.0e-2F*py*pz + 1.0e-1F*pz)*src[time][p_src];
          #pragma omp atomic update
          tau_yy[t1][ii_src_0 + 12][ii_src_1 + 12][ii_src_3 + 12] += r17;
        }
        if (ii_src_0 >= x_m - 1 && ii_src_2 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_0 <= x_M + 1 && ii_src_2 <= z_M + 1 && ii_src_4 <= y_M + 1)
        {
          float r18 = dt*(1.0e-3F*px*py*pz - 1.0e-2F*px*py - 1.0e-2F*py*pz + 1.0e-1F*py)*src[time][p_src];
          #pragma omp atomic update
          tau_yy[t1][ii_src_0 + 12][ii_src_4 + 12][ii_src_2 + 12] += r18;
        }
        if (ii_src_0 >= x_m - 1 && ii_src_3 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_0 <= x_M + 1 && ii_src_3 <= z_M + 1 && ii_src_4 <= y_M + 1)
        {
          float r19 = dt*(-1.0e-3F*px*py*pz + 1.0e-2F*py*pz)*src[time][p_src];
          #pragma omp atomic update
          tau_yy[t1][ii_src_0 + 12][ii_src_4 + 12][ii_src_3 + 12] += r19;
        }
        if (ii_src_1 >= y_m - 1 && ii_src_2 >= z_m - 1 && ii_src_5 >= x_m - 1 && ii_src_1 <= y_M + 1 && ii_src_2 <= z_M + 1 && ii_src_5 <= x_M + 1)
        {
          float r20 = dt*(1.0e-3F*px*py*pz - 1.0e-2F*px*py - 1.0e-2F*px*pz + 1.0e-1F*px)*src[time][p_src];
          #pragma omp atomic update
          tau_yy[t1][ii_src_5 + 12][ii_src_1 + 12][ii_src_2 + 12] += r20;
        }
        if (ii_src_1 >= y_m - 1 && ii_src_3 >= z_m - 1 && ii_src_5 >= x_m - 1 && ii_src_1 <= y_M + 1 && ii_src_3 <= z_M + 1 && ii_src_5 <= x_M + 1)
        {
          float r21 = dt*(-1.0e-3F*px*py*pz + 1.0e-2F*px*pz)*src[time][p_src];
          #pragma omp atomic update
          tau_yy[t1][ii_src_5 + 12][ii_src_1 + 12][ii_src_3 + 12] += r21;
        }
        if (ii_src_2 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_5 >= x_m - 1 && ii_src_2 <= z_M + 1 && ii_src_4 <= y_M + 1 && ii_src_5 <= x_M + 1)
        {
          float r22 = dt*(-1.0e-3F*px*py*pz + 1.0e-2F*px*py)*src[time][p_src];
          #pragma omp atomic update
          tau_yy[t1][ii_src_5 + 12][ii_src_4 + 12][ii_src_2 + 12] += r22;
        }
        if (ii_src_3 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_5 >= x_m - 1 && ii_src_3 <= z_M + 1 && ii_src_4 <= y_M + 1 && ii_src_5 <= x_M + 1)
        {
          float r23 = 1.0e-3F*px*py*pz*dt*src[time][p_src];
          #pragma omp atomic update
          tau_yy[t1][ii_src_5 + 12][ii_src_4 + 12][ii_src_3 + 12] += r23;
        }
      }
    }
    /* End section1 */
    gettimeofday(&end_section1, NULL);
    timers->section1 += (double)(end_section1.tv_sec-start_section1.tv_sec)+(double)(end_section1.tv_usec-start_section1.tv_usec)/1000000;
    struct timeval start_section2, end_section2;
    gettimeofday(&start_section2, NULL);
    /* Begin section2 */
    #pragma omp parallel num_threads(nthreads_nonaffine)
    {
      int chunk_size = (int)(fmax(1, (1.0F/3.0F)*(p_rec1_M - p_rec1_m + 1)/nthreads_nonaffine));
      #pragma omp for collapse(1) schedule(dynamic,chunk_size)
      for (int p_rec1 = p_rec1_m; p_rec1 <= p_rec1_M; p_rec1 += 1)
      {
        int ii_rec1_0 = (int)(floor(-1.0e-1*o_x + 1.0e-1*rec1_coords[p_rec1][0]));
        int ii_rec1_1 = (int)(floor(-1.0e-1*o_y + 1.0e-1*rec1_coords[p_rec1][1]));
        int ii_rec1_2 = (int)(floor(-1.0e-1*o_z + 1.0e-1*rec1_coords[p_rec1][2]));
        int ii_rec1_3 = (int)(floor(-1.0e-1*o_z + 1.0e-1*rec1_coords[p_rec1][2])) + 1;
        int ii_rec1_4 = (int)(floor(-1.0e-1*o_y + 1.0e-1*rec1_coords[p_rec1][1])) + 1;
        int ii_rec1_5 = (int)(floor(-1.0e-1*o_x + 1.0e-1*rec1_coords[p_rec1][0])) + 1;
        float px = (float)(-o_x - 1.0e+1F*(int)(floor(-1.0e-1F*o_x + 1.0e-1F*rec1_coords[p_rec1][0])) + rec1_coords[p_rec1][0]);
        float py = (float)(-o_y - 1.0e+1F*(int)(floor(-1.0e-1F*o_y + 1.0e-1F*rec1_coords[p_rec1][1])) + rec1_coords[p_rec1][1]);
        float pz = (float)(-o_z - 1.0e+1F*(int)(floor(-1.0e-1F*o_z + 1.0e-1F*rec1_coords[p_rec1][2])) + rec1_coords[p_rec1][2]);
        float sum = 0.0F;
        if (ii_rec1_0 >= x_m - 1 && ii_rec1_1 >= y_m - 1 && ii_rec1_2 >= z_m - 1 && ii_rec1_0 <= x_M + 1 && ii_rec1_1 <= y_M + 1 && ii_rec1_2 <= z_M + 1)
        {
          sum += (-1.0e-3F*px*py*pz + 1.0e-2F*px*py + 1.0e-2F*px*pz - 1.0e-1F*px + 1.0e-2F*py*pz - 1.0e-1F*py - 1.0e-1F*pz + 1)*tau_zz[t0][ii_rec1_0 + 12][ii_rec1_1 + 12][ii_rec1_2 + 12];
        }
        if (ii_rec1_0 >= x_m - 1 && ii_rec1_1 >= y_m - 1 && ii_rec1_3 >= z_m - 1 && ii_rec1_0 <= x_M + 1 && ii_rec1_1 <= y_M + 1 && ii_rec1_3 <= z_M + 1)
        {
          sum += (1.0e-3F*px*py*pz - 1.0e-2F*px*pz - 1.0e-2F*py*pz + 1.0e-1F*pz)*tau_zz[t0][ii_rec1_0 + 12][ii_rec1_1 + 12][ii_rec1_3 + 12];
        }
        if (ii_rec1_0 >= x_m - 1 && ii_rec1_2 >= z_m - 1 && ii_rec1_4 >= y_m - 1 && ii_rec1_0 <= x_M + 1 && ii_rec1_2 <= z_M + 1 && ii_rec1_4 <= y_M + 1)
        {
          sum += (1.0e-3F*px*py*pz - 1.0e-2F*px*py - 1.0e-2F*py*pz + 1.0e-1F*py)*tau_zz[t0][ii_rec1_0 + 12][ii_rec1_4 + 12][ii_rec1_2 + 12];
        }
        if (ii_rec1_0 >= x_m - 1 && ii_rec1_3 >= z_m - 1 && ii_rec1_4 >= y_m - 1 && ii_rec1_0 <= x_M + 1 && ii_rec1_3 <= z_M + 1 && ii_rec1_4 <= y_M + 1)
        {
          sum += (-1.0e-3F*px*py*pz + 1.0e-2F*py*pz)*tau_zz[t0][ii_rec1_0 + 12][ii_rec1_4 + 12][ii_rec1_3 + 12];
        }
        if (ii_rec1_1 >= y_m - 1 && ii_rec1_2 >= z_m - 1 && ii_rec1_5 >= x_m - 1 && ii_rec1_1 <= y_M + 1 && ii_rec1_2 <= z_M + 1 && ii_rec1_5 <= x_M + 1)
        {
          sum += (1.0e-3F*px*py*pz - 1.0e-2F*px*py - 1.0e-2F*px*pz + 1.0e-1F*px)*tau_zz[t0][ii_rec1_5 + 12][ii_rec1_1 + 12][ii_rec1_2 + 12];
        }
        if (ii_rec1_1 >= y_m - 1 && ii_rec1_3 >= z_m - 1 && ii_rec1_5 >= x_m - 1 && ii_rec1_1 <= y_M + 1 && ii_rec1_3 <= z_M + 1 && ii_rec1_5 <= x_M + 1)
        {
          sum += (-1.0e-3F*px*py*pz + 1.0e-2F*px*pz)*tau_zz[t0][ii_rec1_5 + 12][ii_rec1_1 + 12][ii_rec1_3 + 12];
        }
        if (ii_rec1_2 >= z_m - 1 && ii_rec1_4 >= y_m - 1 && ii_rec1_5 >= x_m - 1 && ii_rec1_2 <= z_M + 1 && ii_rec1_4 <= y_M + 1 && ii_rec1_5 <= x_M + 1)
        {
          sum += (-1.0e-3F*px*py*pz + 1.0e-2F*px*py)*tau_zz[t0][ii_rec1_5 + 12][ii_rec1_4 + 12][ii_rec1_2 + 12];
        }
        if (ii_rec1_3 >= z_m - 1 && ii_rec1_4 >= y_m - 1 && ii_rec1_5 >= x_m - 1 && ii_rec1_3 <= z_M + 1 && ii_rec1_4 <= y_M + 1 && ii_rec1_5 <= x_M + 1)
        {
          sum += 1.0e-3F*px*py*pz*tau_zz[t0][ii_rec1_5 + 12][ii_rec1_4 + 12][ii_rec1_3 + 12];
        }
        rec1[time][p_rec1] = sum;
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
      int chunk_size = (int)(fmax(1, (1.0F/3.0F)*(p_rec2_M - p_rec2_m + 1)/nthreads_nonaffine));
      #pragma omp for collapse(1) schedule(dynamic,chunk_size)
      for (int p_rec2 = p_rec2_m; p_rec2 <= p_rec2_M; p_rec2 += 1)
      {
        int ii_rec2_0 = (int)(floor(-1.0e-1*o_x + 1.0e-1*rec2_coords[p_rec2][0]));
        int ii_rec2_1 = (int)(floor(-1.0e-1*o_y + 1.0e-1*rec2_coords[p_rec2][1]));
        int ii_rec2_2 = (int)(floor(-1.0e-1*o_z + 1.0e-1*rec2_coords[p_rec2][2]));
        int ii_rec2_3 = (int)(floor(-1.0e-1*o_z + 1.0e-1*rec2_coords[p_rec2][2])) + 1;
        int ii_rec2_4 = (int)(floor(-1.0e-1*o_y + 1.0e-1*rec2_coords[p_rec2][1])) + 1;
        int ii_rec2_5 = (int)(floor(-1.0e-1*o_x + 1.0e-1*rec2_coords[p_rec2][0])) + 1;
        float px = (float)(-o_x - 1.0e+1F*(int)(floor(-1.0e-1F*o_x + 1.0e-1F*rec2_coords[p_rec2][0])) + rec2_coords[p_rec2][0]);
        float py = (float)(-o_y - 1.0e+1F*(int)(floor(-1.0e-1F*o_y + 1.0e-1F*rec2_coords[p_rec2][1])) + rec2_coords[p_rec2][1]);
        float pz = (float)(-o_z - 1.0e+1F*(int)(floor(-1.0e-1F*o_z + 1.0e-1F*rec2_coords[p_rec2][2])) + rec2_coords[p_rec2][2]);
        float sum = 0.0F;
        if (ii_rec2_0 >= x_m - 1 && ii_rec2_1 >= y_m - 1 && ii_rec2_2 >= z_m - 1 && ii_rec2_0 <= x_M + 1 && ii_rec2_1 <= y_M + 1 && ii_rec2_2 <= z_M + 1)
        {
          sum += (-1.0e-3F*px*py*pz + 1.0e-2F*px*py + 1.0e-2F*px*pz - 1.0e-1F*px + 1.0e-2F*py*pz - 1.0e-1F*py - 1.0e-1F*pz + 1)*(1.80375183e-5F*v_x[t0][ii_rec2_0 + 6][ii_rec2_1 + 12][ii_rec2_2 + 12] - 2.59740264e-4F*v_x[t0][ii_rec2_0 + 7][ii_rec2_1 + 12][ii_rec2_2 + 12] + 1.78571431e-3F*v_x[t0][ii_rec2_0 + 8][ii_rec2_1 + 12][ii_rec2_2 + 12] - 7.93650805e-3F*v_x[t0][ii_rec2_0 + 9][ii_rec2_1 + 12][ii_rec2_2 + 12] + 2.67857147e-2F*v_x[t0][ii_rec2_0 + 10][ii_rec2_1 + 12][ii_rec2_2 + 12] - 8.5714287e-2F*v_x[t0][ii_rec2_0 + 11][ii_rec2_1 + 12][ii_rec2_2 + 12] + 8.5714287e-2F*v_x[t0][ii_rec2_0 + 13][ii_rec2_1 + 12][ii_rec2_2 + 12] - 2.67857147e-2F*v_x[t0][ii_rec2_0 + 14][ii_rec2_1 + 12][ii_rec2_2 + 12] + 7.93650805e-3F*v_x[t0][ii_rec2_0 + 15][ii_rec2_1 + 12][ii_rec2_2 + 12] - 1.78571431e-3F*v_x[t0][ii_rec2_0 + 16][ii_rec2_1 + 12][ii_rec2_2 + 12] + 2.59740264e-4F*v_x[t0][ii_rec2_0 + 17][ii_rec2_1 + 12][ii_rec2_2 + 12] - 1.80375183e-5F*v_x[t0][ii_rec2_0 + 18][ii_rec2_1 + 12][ii_rec2_2 + 12] + 1.80375183e-5F*v_y[t0][ii_rec2_0 + 12][ii_rec2_1 + 6][ii_rec2_2 + 12] - 2.59740264e-4F*v_y[t0][ii_rec2_0 + 12][ii_rec2_1 + 7][ii_rec2_2 + 12] + 1.78571431e-3F*v_y[t0][ii_rec2_0 + 12][ii_rec2_1 + 8][ii_rec2_2 + 12] - 7.93650805e-3F*v_y[t0][ii_rec2_0 + 12][ii_rec2_1 + 9][ii_rec2_2 + 12] + 2.67857147e-2F*v_y[t0][ii_rec2_0 + 12][ii_rec2_1 + 10][ii_rec2_2 + 12] - 8.5714287e-2F*v_y[t0][ii_rec2_0 + 12][ii_rec2_1 + 11][ii_rec2_2 + 12] + 8.5714287e-2F*v_y[t0][ii_rec2_0 + 12][ii_rec2_1 + 13][ii_rec2_2 + 12] - 2.67857147e-2F*v_y[t0][ii_rec2_0 + 12][ii_rec2_1 + 14][ii_rec2_2 + 12] + 7.93650805e-3F*v_y[t0][ii_rec2_0 + 12][ii_rec2_1 + 15][ii_rec2_2 + 12] - 1.78571431e-3F*v_y[t0][ii_rec2_0 + 12][ii_rec2_1 + 16][ii_rec2_2 + 12] + 2.59740264e-4F*v_y[t0][ii_rec2_0 + 12][ii_rec2_1 + 17][ii_rec2_2 + 12] - 1.80375183e-5F*v_y[t0][ii_rec2_0 + 12][ii_rec2_1 + 18][ii_rec2_2 + 12] + 1.80375183e-5F*v_z[t0][ii_rec2_0 + 12][ii_rec2_1 + 12][ii_rec2_2 + 6] - 2.59740264e-4F*v_z[t0][ii_rec2_0 + 12][ii_rec2_1 + 12][ii_rec2_2 + 7] + 1.78571431e-3F*v_z[t0][ii_rec2_0 + 12][ii_rec2_1 + 12][ii_rec2_2 + 8] - 7.93650805e-3F*v_z[t0][ii_rec2_0 + 12][ii_rec2_1 + 12][ii_rec2_2 + 9] + 2.67857147e-2F*v_z[t0][ii_rec2_0 + 12][ii_rec2_1 + 12][ii_rec2_2 + 10] - 8.5714287e-2F*v_z[t0][ii_rec2_0 + 12][ii_rec2_1 + 12][ii_rec2_2 + 11] + 8.5714287e-2F*v_z[t0][ii_rec2_0 + 12][ii_rec2_1 + 12][ii_rec2_2 + 13] - 2.67857147e-2F*v_z[t0][ii_rec2_0 + 12][ii_rec2_1 + 12][ii_rec2_2 + 14] + 7.93650805e-3F*v_z[t0][ii_rec2_0 + 12][ii_rec2_1 + 12][ii_rec2_2 + 15] - 1.78571431e-3F*v_z[t0][ii_rec2_0 + 12][ii_rec2_1 + 12][ii_rec2_2 + 16] + 2.59740264e-4F*v_z[t0][ii_rec2_0 + 12][ii_rec2_1 + 12][ii_rec2_2 + 17] - 1.80375183e-5F*v_z[t0][ii_rec2_0 + 12][ii_rec2_1 + 12][ii_rec2_2 + 18]);
        }
        if (ii_rec2_0 >= x_m - 1 && ii_rec2_1 >= y_m - 1 && ii_rec2_3 >= z_m - 1 && ii_rec2_0 <= x_M + 1 && ii_rec2_1 <= y_M + 1 && ii_rec2_3 <= z_M + 1)
        {
          sum += (1.0e-3F*px*py*pz - 1.0e-2F*px*pz - 1.0e-2F*py*pz + 1.0e-1F*pz)*(1.80375183e-5F*v_x[t0][ii_rec2_0 + 6][ii_rec2_1 + 12][ii_rec2_3 + 12] - 2.59740264e-4F*v_x[t0][ii_rec2_0 + 7][ii_rec2_1 + 12][ii_rec2_3 + 12] + 1.78571431e-3F*v_x[t0][ii_rec2_0 + 8][ii_rec2_1 + 12][ii_rec2_3 + 12] - 7.93650805e-3F*v_x[t0][ii_rec2_0 + 9][ii_rec2_1 + 12][ii_rec2_3 + 12] + 2.67857147e-2F*v_x[t0][ii_rec2_0 + 10][ii_rec2_1 + 12][ii_rec2_3 + 12] - 8.5714287e-2F*v_x[t0][ii_rec2_0 + 11][ii_rec2_1 + 12][ii_rec2_3 + 12] + 8.5714287e-2F*v_x[t0][ii_rec2_0 + 13][ii_rec2_1 + 12][ii_rec2_3 + 12] - 2.67857147e-2F*v_x[t0][ii_rec2_0 + 14][ii_rec2_1 + 12][ii_rec2_3 + 12] + 7.93650805e-3F*v_x[t0][ii_rec2_0 + 15][ii_rec2_1 + 12][ii_rec2_3 + 12] - 1.78571431e-3F*v_x[t0][ii_rec2_0 + 16][ii_rec2_1 + 12][ii_rec2_3 + 12] + 2.59740264e-4F*v_x[t0][ii_rec2_0 + 17][ii_rec2_1 + 12][ii_rec2_3 + 12] - 1.80375183e-5F*v_x[t0][ii_rec2_0 + 18][ii_rec2_1 + 12][ii_rec2_3 + 12] + 1.80375183e-5F*v_y[t0][ii_rec2_0 + 12][ii_rec2_1 + 6][ii_rec2_3 + 12] - 2.59740264e-4F*v_y[t0][ii_rec2_0 + 12][ii_rec2_1 + 7][ii_rec2_3 + 12] + 1.78571431e-3F*v_y[t0][ii_rec2_0 + 12][ii_rec2_1 + 8][ii_rec2_3 + 12] - 7.93650805e-3F*v_y[t0][ii_rec2_0 + 12][ii_rec2_1 + 9][ii_rec2_3 + 12] + 2.67857147e-2F*v_y[t0][ii_rec2_0 + 12][ii_rec2_1 + 10][ii_rec2_3 + 12] - 8.5714287e-2F*v_y[t0][ii_rec2_0 + 12][ii_rec2_1 + 11][ii_rec2_3 + 12] + 8.5714287e-2F*v_y[t0][ii_rec2_0 + 12][ii_rec2_1 + 13][ii_rec2_3 + 12] - 2.67857147e-2F*v_y[t0][ii_rec2_0 + 12][ii_rec2_1 + 14][ii_rec2_3 + 12] + 7.93650805e-3F*v_y[t0][ii_rec2_0 + 12][ii_rec2_1 + 15][ii_rec2_3 + 12] - 1.78571431e-3F*v_y[t0][ii_rec2_0 + 12][ii_rec2_1 + 16][ii_rec2_3 + 12] + 2.59740264e-4F*v_y[t0][ii_rec2_0 + 12][ii_rec2_1 + 17][ii_rec2_3 + 12] - 1.80375183e-5F*v_y[t0][ii_rec2_0 + 12][ii_rec2_1 + 18][ii_rec2_3 + 12] + 1.80375183e-5F*v_z[t0][ii_rec2_0 + 12][ii_rec2_1 + 12][ii_rec2_3 + 6] - 2.59740264e-4F*v_z[t0][ii_rec2_0 + 12][ii_rec2_1 + 12][ii_rec2_3 + 7] + 1.78571431e-3F*v_z[t0][ii_rec2_0 + 12][ii_rec2_1 + 12][ii_rec2_3 + 8] - 7.93650805e-3F*v_z[t0][ii_rec2_0 + 12][ii_rec2_1 + 12][ii_rec2_3 + 9] + 2.67857147e-2F*v_z[t0][ii_rec2_0 + 12][ii_rec2_1 + 12][ii_rec2_3 + 10] - 8.5714287e-2F*v_z[t0][ii_rec2_0 + 12][ii_rec2_1 + 12][ii_rec2_3 + 11] + 8.5714287e-2F*v_z[t0][ii_rec2_0 + 12][ii_rec2_1 + 12][ii_rec2_3 + 13] - 2.67857147e-2F*v_z[t0][ii_rec2_0 + 12][ii_rec2_1 + 12][ii_rec2_3 + 14] + 7.93650805e-3F*v_z[t0][ii_rec2_0 + 12][ii_rec2_1 + 12][ii_rec2_3 + 15] - 1.78571431e-3F*v_z[t0][ii_rec2_0 + 12][ii_rec2_1 + 12][ii_rec2_3 + 16] + 2.59740264e-4F*v_z[t0][ii_rec2_0 + 12][ii_rec2_1 + 12][ii_rec2_3 + 17] - 1.80375183e-5F*v_z[t0][ii_rec2_0 + 12][ii_rec2_1 + 12][ii_rec2_3 + 18]);
        }
        if (ii_rec2_0 >= x_m - 1 && ii_rec2_2 >= z_m - 1 && ii_rec2_4 >= y_m - 1 && ii_rec2_0 <= x_M + 1 && ii_rec2_2 <= z_M + 1 && ii_rec2_4 <= y_M + 1)
        {
          sum += (1.0e-3F*px*py*pz - 1.0e-2F*px*py - 1.0e-2F*py*pz + 1.0e-1F*py)*(1.80375183e-5F*v_x[t0][ii_rec2_0 + 6][ii_rec2_4 + 12][ii_rec2_2 + 12] - 2.59740264e-4F*v_x[t0][ii_rec2_0 + 7][ii_rec2_4 + 12][ii_rec2_2 + 12] + 1.78571431e-3F*v_x[t0][ii_rec2_0 + 8][ii_rec2_4 + 12][ii_rec2_2 + 12] - 7.93650805e-3F*v_x[t0][ii_rec2_0 + 9][ii_rec2_4 + 12][ii_rec2_2 + 12] + 2.67857147e-2F*v_x[t0][ii_rec2_0 + 10][ii_rec2_4 + 12][ii_rec2_2 + 12] - 8.5714287e-2F*v_x[t0][ii_rec2_0 + 11][ii_rec2_4 + 12][ii_rec2_2 + 12] + 8.5714287e-2F*v_x[t0][ii_rec2_0 + 13][ii_rec2_4 + 12][ii_rec2_2 + 12] - 2.67857147e-2F*v_x[t0][ii_rec2_0 + 14][ii_rec2_4 + 12][ii_rec2_2 + 12] + 7.93650805e-3F*v_x[t0][ii_rec2_0 + 15][ii_rec2_4 + 12][ii_rec2_2 + 12] - 1.78571431e-3F*v_x[t0][ii_rec2_0 + 16][ii_rec2_4 + 12][ii_rec2_2 + 12] + 2.59740264e-4F*v_x[t0][ii_rec2_0 + 17][ii_rec2_4 + 12][ii_rec2_2 + 12] - 1.80375183e-5F*v_x[t0][ii_rec2_0 + 18][ii_rec2_4 + 12][ii_rec2_2 + 12] + 1.80375183e-5F*v_y[t0][ii_rec2_0 + 12][ii_rec2_4 + 6][ii_rec2_2 + 12] - 2.59740264e-4F*v_y[t0][ii_rec2_0 + 12][ii_rec2_4 + 7][ii_rec2_2 + 12] + 1.78571431e-3F*v_y[t0][ii_rec2_0 + 12][ii_rec2_4 + 8][ii_rec2_2 + 12] - 7.93650805e-3F*v_y[t0][ii_rec2_0 + 12][ii_rec2_4 + 9][ii_rec2_2 + 12] + 2.67857147e-2F*v_y[t0][ii_rec2_0 + 12][ii_rec2_4 + 10][ii_rec2_2 + 12] - 8.5714287e-2F*v_y[t0][ii_rec2_0 + 12][ii_rec2_4 + 11][ii_rec2_2 + 12] + 8.5714287e-2F*v_y[t0][ii_rec2_0 + 12][ii_rec2_4 + 13][ii_rec2_2 + 12] - 2.67857147e-2F*v_y[t0][ii_rec2_0 + 12][ii_rec2_4 + 14][ii_rec2_2 + 12] + 7.93650805e-3F*v_y[t0][ii_rec2_0 + 12][ii_rec2_4 + 15][ii_rec2_2 + 12] - 1.78571431e-3F*v_y[t0][ii_rec2_0 + 12][ii_rec2_4 + 16][ii_rec2_2 + 12] + 2.59740264e-4F*v_y[t0][ii_rec2_0 + 12][ii_rec2_4 + 17][ii_rec2_2 + 12] - 1.80375183e-5F*v_y[t0][ii_rec2_0 + 12][ii_rec2_4 + 18][ii_rec2_2 + 12] + 1.80375183e-5F*v_z[t0][ii_rec2_0 + 12][ii_rec2_4 + 12][ii_rec2_2 + 6] - 2.59740264e-4F*v_z[t0][ii_rec2_0 + 12][ii_rec2_4 + 12][ii_rec2_2 + 7] + 1.78571431e-3F*v_z[t0][ii_rec2_0 + 12][ii_rec2_4 + 12][ii_rec2_2 + 8] - 7.93650805e-3F*v_z[t0][ii_rec2_0 + 12][ii_rec2_4 + 12][ii_rec2_2 + 9] + 2.67857147e-2F*v_z[t0][ii_rec2_0 + 12][ii_rec2_4 + 12][ii_rec2_2 + 10] - 8.5714287e-2F*v_z[t0][ii_rec2_0 + 12][ii_rec2_4 + 12][ii_rec2_2 + 11] + 8.5714287e-2F*v_z[t0][ii_rec2_0 + 12][ii_rec2_4 + 12][ii_rec2_2 + 13] - 2.67857147e-2F*v_z[t0][ii_rec2_0 + 12][ii_rec2_4 + 12][ii_rec2_2 + 14] + 7.93650805e-3F*v_z[t0][ii_rec2_0 + 12][ii_rec2_4 + 12][ii_rec2_2 + 15] - 1.78571431e-3F*v_z[t0][ii_rec2_0 + 12][ii_rec2_4 + 12][ii_rec2_2 + 16] + 2.59740264e-4F*v_z[t0][ii_rec2_0 + 12][ii_rec2_4 + 12][ii_rec2_2 + 17] - 1.80375183e-5F*v_z[t0][ii_rec2_0 + 12][ii_rec2_4 + 12][ii_rec2_2 + 18]);
        }
        if (ii_rec2_0 >= x_m - 1 && ii_rec2_3 >= z_m - 1 && ii_rec2_4 >= y_m - 1 && ii_rec2_0 <= x_M + 1 && ii_rec2_3 <= z_M + 1 && ii_rec2_4 <= y_M + 1)
        {
          sum += (-1.0e-3F*px*py*pz + 1.0e-2F*py*pz)*(1.80375183e-5F*v_x[t0][ii_rec2_0 + 6][ii_rec2_4 + 12][ii_rec2_3 + 12] - 2.59740264e-4F*v_x[t0][ii_rec2_0 + 7][ii_rec2_4 + 12][ii_rec2_3 + 12] + 1.78571431e-3F*v_x[t0][ii_rec2_0 + 8][ii_rec2_4 + 12][ii_rec2_3 + 12] - 7.93650805e-3F*v_x[t0][ii_rec2_0 + 9][ii_rec2_4 + 12][ii_rec2_3 + 12] + 2.67857147e-2F*v_x[t0][ii_rec2_0 + 10][ii_rec2_4 + 12][ii_rec2_3 + 12] - 8.5714287e-2F*v_x[t0][ii_rec2_0 + 11][ii_rec2_4 + 12][ii_rec2_3 + 12] + 8.5714287e-2F*v_x[t0][ii_rec2_0 + 13][ii_rec2_4 + 12][ii_rec2_3 + 12] - 2.67857147e-2F*v_x[t0][ii_rec2_0 + 14][ii_rec2_4 + 12][ii_rec2_3 + 12] + 7.93650805e-3F*v_x[t0][ii_rec2_0 + 15][ii_rec2_4 + 12][ii_rec2_3 + 12] - 1.78571431e-3F*v_x[t0][ii_rec2_0 + 16][ii_rec2_4 + 12][ii_rec2_3 + 12] + 2.59740264e-4F*v_x[t0][ii_rec2_0 + 17][ii_rec2_4 + 12][ii_rec2_3 + 12] - 1.80375183e-5F*v_x[t0][ii_rec2_0 + 18][ii_rec2_4 + 12][ii_rec2_3 + 12] + 1.80375183e-5F*v_y[t0][ii_rec2_0 + 12][ii_rec2_4 + 6][ii_rec2_3 + 12] - 2.59740264e-4F*v_y[t0][ii_rec2_0 + 12][ii_rec2_4 + 7][ii_rec2_3 + 12] + 1.78571431e-3F*v_y[t0][ii_rec2_0 + 12][ii_rec2_4 + 8][ii_rec2_3 + 12] - 7.93650805e-3F*v_y[t0][ii_rec2_0 + 12][ii_rec2_4 + 9][ii_rec2_3 + 12] + 2.67857147e-2F*v_y[t0][ii_rec2_0 + 12][ii_rec2_4 + 10][ii_rec2_3 + 12] - 8.5714287e-2F*v_y[t0][ii_rec2_0 + 12][ii_rec2_4 + 11][ii_rec2_3 + 12] + 8.5714287e-2F*v_y[t0][ii_rec2_0 + 12][ii_rec2_4 + 13][ii_rec2_3 + 12] - 2.67857147e-2F*v_y[t0][ii_rec2_0 + 12][ii_rec2_4 + 14][ii_rec2_3 + 12] + 7.93650805e-3F*v_y[t0][ii_rec2_0 + 12][ii_rec2_4 + 15][ii_rec2_3 + 12] - 1.78571431e-3F*v_y[t0][ii_rec2_0 + 12][ii_rec2_4 + 16][ii_rec2_3 + 12] + 2.59740264e-4F*v_y[t0][ii_rec2_0 + 12][ii_rec2_4 + 17][ii_rec2_3 + 12] - 1.80375183e-5F*v_y[t0][ii_rec2_0 + 12][ii_rec2_4 + 18][ii_rec2_3 + 12] + 1.80375183e-5F*v_z[t0][ii_rec2_0 + 12][ii_rec2_4 + 12][ii_rec2_3 + 6] - 2.59740264e-4F*v_z[t0][ii_rec2_0 + 12][ii_rec2_4 + 12][ii_rec2_3 + 7] + 1.78571431e-3F*v_z[t0][ii_rec2_0 + 12][ii_rec2_4 + 12][ii_rec2_3 + 8] - 7.93650805e-3F*v_z[t0][ii_rec2_0 + 12][ii_rec2_4 + 12][ii_rec2_3 + 9] + 2.67857147e-2F*v_z[t0][ii_rec2_0 + 12][ii_rec2_4 + 12][ii_rec2_3 + 10] - 8.5714287e-2F*v_z[t0][ii_rec2_0 + 12][ii_rec2_4 + 12][ii_rec2_3 + 11] + 8.5714287e-2F*v_z[t0][ii_rec2_0 + 12][ii_rec2_4 + 12][ii_rec2_3 + 13] - 2.67857147e-2F*v_z[t0][ii_rec2_0 + 12][ii_rec2_4 + 12][ii_rec2_3 + 14] + 7.93650805e-3F*v_z[t0][ii_rec2_0 + 12][ii_rec2_4 + 12][ii_rec2_3 + 15] - 1.78571431e-3F*v_z[t0][ii_rec2_0 + 12][ii_rec2_4 + 12][ii_rec2_3 + 16] + 2.59740264e-4F*v_z[t0][ii_rec2_0 + 12][ii_rec2_4 + 12][ii_rec2_3 + 17] - 1.80375183e-5F*v_z[t0][ii_rec2_0 + 12][ii_rec2_4 + 12][ii_rec2_3 + 18]);
        }
        if (ii_rec2_1 >= y_m - 1 && ii_rec2_2 >= z_m - 1 && ii_rec2_5 >= x_m - 1 && ii_rec2_1 <= y_M + 1 && ii_rec2_2 <= z_M + 1 && ii_rec2_5 <= x_M + 1)
        {
          sum += (1.0e-3F*px*py*pz - 1.0e-2F*px*py - 1.0e-2F*px*pz + 1.0e-1F*px)*(1.80375183e-5F*v_x[t0][ii_rec2_5 + 6][ii_rec2_1 + 12][ii_rec2_2 + 12] - 2.59740264e-4F*v_x[t0][ii_rec2_5 + 7][ii_rec2_1 + 12][ii_rec2_2 + 12] + 1.78571431e-3F*v_x[t0][ii_rec2_5 + 8][ii_rec2_1 + 12][ii_rec2_2 + 12] - 7.93650805e-3F*v_x[t0][ii_rec2_5 + 9][ii_rec2_1 + 12][ii_rec2_2 + 12] + 2.67857147e-2F*v_x[t0][ii_rec2_5 + 10][ii_rec2_1 + 12][ii_rec2_2 + 12] - 8.5714287e-2F*v_x[t0][ii_rec2_5 + 11][ii_rec2_1 + 12][ii_rec2_2 + 12] + 8.5714287e-2F*v_x[t0][ii_rec2_5 + 13][ii_rec2_1 + 12][ii_rec2_2 + 12] - 2.67857147e-2F*v_x[t0][ii_rec2_5 + 14][ii_rec2_1 + 12][ii_rec2_2 + 12] + 7.93650805e-3F*v_x[t0][ii_rec2_5 + 15][ii_rec2_1 + 12][ii_rec2_2 + 12] - 1.78571431e-3F*v_x[t0][ii_rec2_5 + 16][ii_rec2_1 + 12][ii_rec2_2 + 12] + 2.59740264e-4F*v_x[t0][ii_rec2_5 + 17][ii_rec2_1 + 12][ii_rec2_2 + 12] - 1.80375183e-5F*v_x[t0][ii_rec2_5 + 18][ii_rec2_1 + 12][ii_rec2_2 + 12] + 1.80375183e-5F*v_y[t0][ii_rec2_5 + 12][ii_rec2_1 + 6][ii_rec2_2 + 12] - 2.59740264e-4F*v_y[t0][ii_rec2_5 + 12][ii_rec2_1 + 7][ii_rec2_2 + 12] + 1.78571431e-3F*v_y[t0][ii_rec2_5 + 12][ii_rec2_1 + 8][ii_rec2_2 + 12] - 7.93650805e-3F*v_y[t0][ii_rec2_5 + 12][ii_rec2_1 + 9][ii_rec2_2 + 12] + 2.67857147e-2F*v_y[t0][ii_rec2_5 + 12][ii_rec2_1 + 10][ii_rec2_2 + 12] - 8.5714287e-2F*v_y[t0][ii_rec2_5 + 12][ii_rec2_1 + 11][ii_rec2_2 + 12] + 8.5714287e-2F*v_y[t0][ii_rec2_5 + 12][ii_rec2_1 + 13][ii_rec2_2 + 12] - 2.67857147e-2F*v_y[t0][ii_rec2_5 + 12][ii_rec2_1 + 14][ii_rec2_2 + 12] + 7.93650805e-3F*v_y[t0][ii_rec2_5 + 12][ii_rec2_1 + 15][ii_rec2_2 + 12] - 1.78571431e-3F*v_y[t0][ii_rec2_5 + 12][ii_rec2_1 + 16][ii_rec2_2 + 12] + 2.59740264e-4F*v_y[t0][ii_rec2_5 + 12][ii_rec2_1 + 17][ii_rec2_2 + 12] - 1.80375183e-5F*v_y[t0][ii_rec2_5 + 12][ii_rec2_1 + 18][ii_rec2_2 + 12] + 1.80375183e-5F*v_z[t0][ii_rec2_5 + 12][ii_rec2_1 + 12][ii_rec2_2 + 6] - 2.59740264e-4F*v_z[t0][ii_rec2_5 + 12][ii_rec2_1 + 12][ii_rec2_2 + 7] + 1.78571431e-3F*v_z[t0][ii_rec2_5 + 12][ii_rec2_1 + 12][ii_rec2_2 + 8] - 7.93650805e-3F*v_z[t0][ii_rec2_5 + 12][ii_rec2_1 + 12][ii_rec2_2 + 9] + 2.67857147e-2F*v_z[t0][ii_rec2_5 + 12][ii_rec2_1 + 12][ii_rec2_2 + 10] - 8.5714287e-2F*v_z[t0][ii_rec2_5 + 12][ii_rec2_1 + 12][ii_rec2_2 + 11] + 8.5714287e-2F*v_z[t0][ii_rec2_5 + 12][ii_rec2_1 + 12][ii_rec2_2 + 13] - 2.67857147e-2F*v_z[t0][ii_rec2_5 + 12][ii_rec2_1 + 12][ii_rec2_2 + 14] + 7.93650805e-3F*v_z[t0][ii_rec2_5 + 12][ii_rec2_1 + 12][ii_rec2_2 + 15] - 1.78571431e-3F*v_z[t0][ii_rec2_5 + 12][ii_rec2_1 + 12][ii_rec2_2 + 16] + 2.59740264e-4F*v_z[t0][ii_rec2_5 + 12][ii_rec2_1 + 12][ii_rec2_2 + 17] - 1.80375183e-5F*v_z[t0][ii_rec2_5 + 12][ii_rec2_1 + 12][ii_rec2_2 + 18]);
        }
        if (ii_rec2_1 >= y_m - 1 && ii_rec2_3 >= z_m - 1 && ii_rec2_5 >= x_m - 1 && ii_rec2_1 <= y_M + 1 && ii_rec2_3 <= z_M + 1 && ii_rec2_5 <= x_M + 1)
        {
          sum += (-1.0e-3F*px*py*pz + 1.0e-2F*px*pz)*(1.80375183e-5F*v_x[t0][ii_rec2_5 + 6][ii_rec2_1 + 12][ii_rec2_3 + 12] - 2.59740264e-4F*v_x[t0][ii_rec2_5 + 7][ii_rec2_1 + 12][ii_rec2_3 + 12] + 1.78571431e-3F*v_x[t0][ii_rec2_5 + 8][ii_rec2_1 + 12][ii_rec2_3 + 12] - 7.93650805e-3F*v_x[t0][ii_rec2_5 + 9][ii_rec2_1 + 12][ii_rec2_3 + 12] + 2.67857147e-2F*v_x[t0][ii_rec2_5 + 10][ii_rec2_1 + 12][ii_rec2_3 + 12] - 8.5714287e-2F*v_x[t0][ii_rec2_5 + 11][ii_rec2_1 + 12][ii_rec2_3 + 12] + 8.5714287e-2F*v_x[t0][ii_rec2_5 + 13][ii_rec2_1 + 12][ii_rec2_3 + 12] - 2.67857147e-2F*v_x[t0][ii_rec2_5 + 14][ii_rec2_1 + 12][ii_rec2_3 + 12] + 7.93650805e-3F*v_x[t0][ii_rec2_5 + 15][ii_rec2_1 + 12][ii_rec2_3 + 12] - 1.78571431e-3F*v_x[t0][ii_rec2_5 + 16][ii_rec2_1 + 12][ii_rec2_3 + 12] + 2.59740264e-4F*v_x[t0][ii_rec2_5 + 17][ii_rec2_1 + 12][ii_rec2_3 + 12] - 1.80375183e-5F*v_x[t0][ii_rec2_5 + 18][ii_rec2_1 + 12][ii_rec2_3 + 12] + 1.80375183e-5F*v_y[t0][ii_rec2_5 + 12][ii_rec2_1 + 6][ii_rec2_3 + 12] - 2.59740264e-4F*v_y[t0][ii_rec2_5 + 12][ii_rec2_1 + 7][ii_rec2_3 + 12] + 1.78571431e-3F*v_y[t0][ii_rec2_5 + 12][ii_rec2_1 + 8][ii_rec2_3 + 12] - 7.93650805e-3F*v_y[t0][ii_rec2_5 + 12][ii_rec2_1 + 9][ii_rec2_3 + 12] + 2.67857147e-2F*v_y[t0][ii_rec2_5 + 12][ii_rec2_1 + 10][ii_rec2_3 + 12] - 8.5714287e-2F*v_y[t0][ii_rec2_5 + 12][ii_rec2_1 + 11][ii_rec2_3 + 12] + 8.5714287e-2F*v_y[t0][ii_rec2_5 + 12][ii_rec2_1 + 13][ii_rec2_3 + 12] - 2.67857147e-2F*v_y[t0][ii_rec2_5 + 12][ii_rec2_1 + 14][ii_rec2_3 + 12] + 7.93650805e-3F*v_y[t0][ii_rec2_5 + 12][ii_rec2_1 + 15][ii_rec2_3 + 12] - 1.78571431e-3F*v_y[t0][ii_rec2_5 + 12][ii_rec2_1 + 16][ii_rec2_3 + 12] + 2.59740264e-4F*v_y[t0][ii_rec2_5 + 12][ii_rec2_1 + 17][ii_rec2_3 + 12] - 1.80375183e-5F*v_y[t0][ii_rec2_5 + 12][ii_rec2_1 + 18][ii_rec2_3 + 12] + 1.80375183e-5F*v_z[t0][ii_rec2_5 + 12][ii_rec2_1 + 12][ii_rec2_3 + 6] - 2.59740264e-4F*v_z[t0][ii_rec2_5 + 12][ii_rec2_1 + 12][ii_rec2_3 + 7] + 1.78571431e-3F*v_z[t0][ii_rec2_5 + 12][ii_rec2_1 + 12][ii_rec2_3 + 8] - 7.93650805e-3F*v_z[t0][ii_rec2_5 + 12][ii_rec2_1 + 12][ii_rec2_3 + 9] + 2.67857147e-2F*v_z[t0][ii_rec2_5 + 12][ii_rec2_1 + 12][ii_rec2_3 + 10] - 8.5714287e-2F*v_z[t0][ii_rec2_5 + 12][ii_rec2_1 + 12][ii_rec2_3 + 11] + 8.5714287e-2F*v_z[t0][ii_rec2_5 + 12][ii_rec2_1 + 12][ii_rec2_3 + 13] - 2.67857147e-2F*v_z[t0][ii_rec2_5 + 12][ii_rec2_1 + 12][ii_rec2_3 + 14] + 7.93650805e-3F*v_z[t0][ii_rec2_5 + 12][ii_rec2_1 + 12][ii_rec2_3 + 15] - 1.78571431e-3F*v_z[t0][ii_rec2_5 + 12][ii_rec2_1 + 12][ii_rec2_3 + 16] + 2.59740264e-4F*v_z[t0][ii_rec2_5 + 12][ii_rec2_1 + 12][ii_rec2_3 + 17] - 1.80375183e-5F*v_z[t0][ii_rec2_5 + 12][ii_rec2_1 + 12][ii_rec2_3 + 18]);
        }
        if (ii_rec2_2 >= z_m - 1 && ii_rec2_4 >= y_m - 1 && ii_rec2_5 >= x_m - 1 && ii_rec2_2 <= z_M + 1 && ii_rec2_4 <= y_M + 1 && ii_rec2_5 <= x_M + 1)
        {
          sum += (-1.0e-3F*px*py*pz + 1.0e-2F*px*py)*(1.80375183e-5F*v_x[t0][ii_rec2_5 + 6][ii_rec2_4 + 12][ii_rec2_2 + 12] - 2.59740264e-4F*v_x[t0][ii_rec2_5 + 7][ii_rec2_4 + 12][ii_rec2_2 + 12] + 1.78571431e-3F*v_x[t0][ii_rec2_5 + 8][ii_rec2_4 + 12][ii_rec2_2 + 12] - 7.93650805e-3F*v_x[t0][ii_rec2_5 + 9][ii_rec2_4 + 12][ii_rec2_2 + 12] + 2.67857147e-2F*v_x[t0][ii_rec2_5 + 10][ii_rec2_4 + 12][ii_rec2_2 + 12] - 8.5714287e-2F*v_x[t0][ii_rec2_5 + 11][ii_rec2_4 + 12][ii_rec2_2 + 12] + 8.5714287e-2F*v_x[t0][ii_rec2_5 + 13][ii_rec2_4 + 12][ii_rec2_2 + 12] - 2.67857147e-2F*v_x[t0][ii_rec2_5 + 14][ii_rec2_4 + 12][ii_rec2_2 + 12] + 7.93650805e-3F*v_x[t0][ii_rec2_5 + 15][ii_rec2_4 + 12][ii_rec2_2 + 12] - 1.78571431e-3F*v_x[t0][ii_rec2_5 + 16][ii_rec2_4 + 12][ii_rec2_2 + 12] + 2.59740264e-4F*v_x[t0][ii_rec2_5 + 17][ii_rec2_4 + 12][ii_rec2_2 + 12] - 1.80375183e-5F*v_x[t0][ii_rec2_5 + 18][ii_rec2_4 + 12][ii_rec2_2 + 12] + 1.80375183e-5F*v_y[t0][ii_rec2_5 + 12][ii_rec2_4 + 6][ii_rec2_2 + 12] - 2.59740264e-4F*v_y[t0][ii_rec2_5 + 12][ii_rec2_4 + 7][ii_rec2_2 + 12] + 1.78571431e-3F*v_y[t0][ii_rec2_5 + 12][ii_rec2_4 + 8][ii_rec2_2 + 12] - 7.93650805e-3F*v_y[t0][ii_rec2_5 + 12][ii_rec2_4 + 9][ii_rec2_2 + 12] + 2.67857147e-2F*v_y[t0][ii_rec2_5 + 12][ii_rec2_4 + 10][ii_rec2_2 + 12] - 8.5714287e-2F*v_y[t0][ii_rec2_5 + 12][ii_rec2_4 + 11][ii_rec2_2 + 12] + 8.5714287e-2F*v_y[t0][ii_rec2_5 + 12][ii_rec2_4 + 13][ii_rec2_2 + 12] - 2.67857147e-2F*v_y[t0][ii_rec2_5 + 12][ii_rec2_4 + 14][ii_rec2_2 + 12] + 7.93650805e-3F*v_y[t0][ii_rec2_5 + 12][ii_rec2_4 + 15][ii_rec2_2 + 12] - 1.78571431e-3F*v_y[t0][ii_rec2_5 + 12][ii_rec2_4 + 16][ii_rec2_2 + 12] + 2.59740264e-4F*v_y[t0][ii_rec2_5 + 12][ii_rec2_4 + 17][ii_rec2_2 + 12] - 1.80375183e-5F*v_y[t0][ii_rec2_5 + 12][ii_rec2_4 + 18][ii_rec2_2 + 12] + 1.80375183e-5F*v_z[t0][ii_rec2_5 + 12][ii_rec2_4 + 12][ii_rec2_2 + 6] - 2.59740264e-4F*v_z[t0][ii_rec2_5 + 12][ii_rec2_4 + 12][ii_rec2_2 + 7] + 1.78571431e-3F*v_z[t0][ii_rec2_5 + 12][ii_rec2_4 + 12][ii_rec2_2 + 8] - 7.93650805e-3F*v_z[t0][ii_rec2_5 + 12][ii_rec2_4 + 12][ii_rec2_2 + 9] + 2.67857147e-2F*v_z[t0][ii_rec2_5 + 12][ii_rec2_4 + 12][ii_rec2_2 + 10] - 8.5714287e-2F*v_z[t0][ii_rec2_5 + 12][ii_rec2_4 + 12][ii_rec2_2 + 11] + 8.5714287e-2F*v_z[t0][ii_rec2_5 + 12][ii_rec2_4 + 12][ii_rec2_2 + 13] - 2.67857147e-2F*v_z[t0][ii_rec2_5 + 12][ii_rec2_4 + 12][ii_rec2_2 + 14] + 7.93650805e-3F*v_z[t0][ii_rec2_5 + 12][ii_rec2_4 + 12][ii_rec2_2 + 15] - 1.78571431e-3F*v_z[t0][ii_rec2_5 + 12][ii_rec2_4 + 12][ii_rec2_2 + 16] + 2.59740264e-4F*v_z[t0][ii_rec2_5 + 12][ii_rec2_4 + 12][ii_rec2_2 + 17] - 1.80375183e-5F*v_z[t0][ii_rec2_5 + 12][ii_rec2_4 + 12][ii_rec2_2 + 18]);
        }
        if (ii_rec2_3 >= z_m - 1 && ii_rec2_4 >= y_m - 1 && ii_rec2_5 >= x_m - 1 && ii_rec2_3 <= z_M + 1 && ii_rec2_4 <= y_M + 1 && ii_rec2_5 <= x_M + 1)
        {
          sum += 1.0e-3F*px*py*pz*(1.80375183e-5F*v_x[t0][ii_rec2_5 + 6][ii_rec2_4 + 12][ii_rec2_3 + 12] - 2.59740264e-4F*v_x[t0][ii_rec2_5 + 7][ii_rec2_4 + 12][ii_rec2_3 + 12] + 1.78571431e-3F*v_x[t0][ii_rec2_5 + 8][ii_rec2_4 + 12][ii_rec2_3 + 12] - 7.93650805e-3F*v_x[t0][ii_rec2_5 + 9][ii_rec2_4 + 12][ii_rec2_3 + 12] + 2.67857147e-2F*v_x[t0][ii_rec2_5 + 10][ii_rec2_4 + 12][ii_rec2_3 + 12] - 8.5714287e-2F*v_x[t0][ii_rec2_5 + 11][ii_rec2_4 + 12][ii_rec2_3 + 12] + 8.5714287e-2F*v_x[t0][ii_rec2_5 + 13][ii_rec2_4 + 12][ii_rec2_3 + 12] - 2.67857147e-2F*v_x[t0][ii_rec2_5 + 14][ii_rec2_4 + 12][ii_rec2_3 + 12] + 7.93650805e-3F*v_x[t0][ii_rec2_5 + 15][ii_rec2_4 + 12][ii_rec2_3 + 12] - 1.78571431e-3F*v_x[t0][ii_rec2_5 + 16][ii_rec2_4 + 12][ii_rec2_3 + 12] + 2.59740264e-4F*v_x[t0][ii_rec2_5 + 17][ii_rec2_4 + 12][ii_rec2_3 + 12] - 1.80375183e-5F*v_x[t0][ii_rec2_5 + 18][ii_rec2_4 + 12][ii_rec2_3 + 12] + 1.80375183e-5F*v_y[t0][ii_rec2_5 + 12][ii_rec2_4 + 6][ii_rec2_3 + 12] - 2.59740264e-4F*v_y[t0][ii_rec2_5 + 12][ii_rec2_4 + 7][ii_rec2_3 + 12] + 1.78571431e-3F*v_y[t0][ii_rec2_5 + 12][ii_rec2_4 + 8][ii_rec2_3 + 12] - 7.93650805e-3F*v_y[t0][ii_rec2_5 + 12][ii_rec2_4 + 9][ii_rec2_3 + 12] + 2.67857147e-2F*v_y[t0][ii_rec2_5 + 12][ii_rec2_4 + 10][ii_rec2_3 + 12] - 8.5714287e-2F*v_y[t0][ii_rec2_5 + 12][ii_rec2_4 + 11][ii_rec2_3 + 12] + 8.5714287e-2F*v_y[t0][ii_rec2_5 + 12][ii_rec2_4 + 13][ii_rec2_3 + 12] - 2.67857147e-2F*v_y[t0][ii_rec2_5 + 12][ii_rec2_4 + 14][ii_rec2_3 + 12] + 7.93650805e-3F*v_y[t0][ii_rec2_5 + 12][ii_rec2_4 + 15][ii_rec2_3 + 12] - 1.78571431e-3F*v_y[t0][ii_rec2_5 + 12][ii_rec2_4 + 16][ii_rec2_3 + 12] + 2.59740264e-4F*v_y[t0][ii_rec2_5 + 12][ii_rec2_4 + 17][ii_rec2_3 + 12] - 1.80375183e-5F*v_y[t0][ii_rec2_5 + 12][ii_rec2_4 + 18][ii_rec2_3 + 12] + 1.80375183e-5F*v_z[t0][ii_rec2_5 + 12][ii_rec2_4 + 12][ii_rec2_3 + 6] - 2.59740264e-4F*v_z[t0][ii_rec2_5 + 12][ii_rec2_4 + 12][ii_rec2_3 + 7] + 1.78571431e-3F*v_z[t0][ii_rec2_5 + 12][ii_rec2_4 + 12][ii_rec2_3 + 8] - 7.93650805e-3F*v_z[t0][ii_rec2_5 + 12][ii_rec2_4 + 12][ii_rec2_3 + 9] + 2.67857147e-2F*v_z[t0][ii_rec2_5 + 12][ii_rec2_4 + 12][ii_rec2_3 + 10] - 8.5714287e-2F*v_z[t0][ii_rec2_5 + 12][ii_rec2_4 + 12][ii_rec2_3 + 11] + 8.5714287e-2F*v_z[t0][ii_rec2_5 + 12][ii_rec2_4 + 12][ii_rec2_3 + 13] - 2.67857147e-2F*v_z[t0][ii_rec2_5 + 12][ii_rec2_4 + 12][ii_rec2_3 + 14] + 7.93650805e-3F*v_z[t0][ii_rec2_5 + 12][ii_rec2_4 + 12][ii_rec2_3 + 15] - 1.78571431e-3F*v_z[t0][ii_rec2_5 + 12][ii_rec2_4 + 12][ii_rec2_3 + 16] + 2.59740264e-4F*v_z[t0][ii_rec2_5 + 12][ii_rec2_4 + 12][ii_rec2_3 + 17] - 1.80375183e-5F*v_z[t0][ii_rec2_5 + 12][ii_rec2_4 + 12][ii_rec2_3 + 18]);
        }
        rec2[time][p_rec2] = sum;
      }
    }
    /* End section3 */
    gettimeofday(&end_section3, NULL);
    timers->section3 += (double)(end_section3.tv_sec-start_section3.tv_sec)+(double)(end_section3.tv_usec-start_section3.tv_usec)/1000000;
  }
  return 0;
}

void bf0(struct dataobj *restrict damp_vec, struct dataobj *restrict irho_vec, struct dataobj *restrict tau_xx_vec, struct dataobj *restrict tau_xy_vec, struct dataobj *restrict tau_xz_vec, struct dataobj *restrict tau_yy_vec, struct dataobj *restrict tau_yz_vec, struct dataobj *restrict tau_zz_vec, struct dataobj *restrict v_x_vec, struct dataobj *restrict v_y_vec, struct dataobj *restrict v_z_vec, const int t0, const int t1, const int x0_blk0_size, const int x_M, const int x_m, const int y0_blk0_size, const int y_M, const int y_m, const int z_M, const int z_m, const int nthreads)
{
  float (*restrict damp)[damp_vec->size[1]][damp_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[damp_vec->size[1]][damp_vec->size[2]]) damp_vec->data;
  float (*restrict irho)[irho_vec->size[1]][irho_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[irho_vec->size[1]][irho_vec->size[2]]) irho_vec->data;
  float (*restrict tau_xx)[tau_xx_vec->size[1]][tau_xx_vec->size[2]][tau_xx_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[tau_xx_vec->size[1]][tau_xx_vec->size[2]][tau_xx_vec->size[3]]) tau_xx_vec->data;
  float (*restrict tau_xy)[tau_xy_vec->size[1]][tau_xy_vec->size[2]][tau_xy_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[tau_xy_vec->size[1]][tau_xy_vec->size[2]][tau_xy_vec->size[3]]) tau_xy_vec->data;
  float (*restrict tau_xz)[tau_xz_vec->size[1]][tau_xz_vec->size[2]][tau_xz_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[tau_xz_vec->size[1]][tau_xz_vec->size[2]][tau_xz_vec->size[3]]) tau_xz_vec->data;
  float (*restrict tau_yy)[tau_yy_vec->size[1]][tau_yy_vec->size[2]][tau_yy_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[tau_yy_vec->size[1]][tau_yy_vec->size[2]][tau_yy_vec->size[3]]) tau_yy_vec->data;
  float (*restrict tau_yz)[tau_yz_vec->size[1]][tau_yz_vec->size[2]][tau_yz_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[tau_yz_vec->size[1]][tau_yz_vec->size[2]][tau_yz_vec->size[3]]) tau_yz_vec->data;
  float (*restrict tau_zz)[tau_zz_vec->size[1]][tau_zz_vec->size[2]][tau_zz_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[tau_zz_vec->size[1]][tau_zz_vec->size[2]][tau_zz_vec->size[3]]) tau_zz_vec->data;
  float (*restrict v_x)[v_x_vec->size[1]][v_x_vec->size[2]][v_x_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[v_x_vec->size[1]][v_x_vec->size[2]][v_x_vec->size[3]]) v_x_vec->data;
  float (*restrict v_y)[v_y_vec->size[1]][v_y_vec->size[2]][v_y_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[v_y_vec->size[1]][v_y_vec->size[2]][v_y_vec->size[3]]) v_y_vec->data;
  float (*restrict v_z)[v_z_vec->size[1]][v_z_vec->size[2]][v_z_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[v_z_vec->size[1]][v_z_vec->size[2]][v_z_vec->size[3]]) v_z_vec->data;
  if (x0_blk0_size == 0)
  {
    return;
  }
  #pragma omp parallel num_threads(nthreads)
  {
    #pragma omp for collapse(1) schedule(dynamic,1)
    for (int x0_blk0 = x_m; x0_blk0 <= x_M; x0_blk0 += x0_blk0_size)
    {
      for (int y0_blk0 = y_m; y0_blk0 <= y_M; y0_blk0 += y0_blk0_size)
      {
        for (int x = x0_blk0; x <= x0_blk0 + x0_blk0_size - 1; x += 1)
        {
          for (int y = y0_blk0; y <= y0_blk0 + y0_blk0_size - 1; y += 1)
          {
            #pragma omp simd aligned(damp,irho,tau_xx,tau_xy,tau_xz,tau_yy,tau_yz,tau_zz,v_x,v_y,v_z:32)
            for (int z = z_m; z <= z_M; z += 1)
            {
              v_x[t1][x + 12][y + 12][z + 12] = 7.00999975204468e-1F*(irho[x + 12][y + 12][z + 12] + irho[x + 13][y + 12][z + 12])*(2.18478119e-6F*(tau_xx[t0][x + 7][y + 12][z + 12] - tau_xx[t0][x + 18][y + 12][z + 12] + tau_xy[t0][x + 12][y + 6][z + 12] - tau_xy[t0][x + 12][y + 17][z + 12] + tau_xz[t0][x + 12][y + 12][z + 6] - tau_xz[t0][x + 12][y + 12][z + 17]) + 3.59005404e-5F*(-tau_xx[t0][x + 8][y + 12][z + 12] + tau_xx[t0][x + 17][y + 12][z + 12] - tau_xy[t0][x + 12][y + 7][z + 12] + tau_xy[t0][x + 12][y + 16][z + 12] - tau_xz[t0][x + 12][y + 12][z + 7] + tau_xz[t0][x + 12][y + 12][z + 16]) + 2.96728956e-4F*(tau_xx[t0][x + 9][y + 12][z + 12] - tau_xx[t0][x + 16][y + 12][z + 12] + tau_xy[t0][x + 12][y + 8][z + 12] - tau_xy[t0][x + 12][y + 15][z + 12] + tau_xz[t0][x + 12][y + 12][z + 8] - tau_xz[t0][x + 12][y + 12][z + 15]) + 1.74476626e-3F*(-tau_xx[t0][x + 10][y + 12][z + 12] + tau_xx[t0][x + 15][y + 12][z + 12] - tau_xy[t0][x + 12][y + 9][z + 12] + tau_xy[t0][x + 12][y + 14][z + 12] - tau_xz[t0][x + 12][y + 12][z + 9] + tau_xz[t0][x + 12][y + 12][z + 14]) + 9.6931459e-3F*(tau_xx[t0][x + 11][y + 12][z + 12] - tau_xx[t0][x + 14][y + 12][z + 12] + tau_xy[t0][x + 12][y + 10][z + 12] - tau_xy[t0][x + 12][y + 13][z + 12] + tau_xz[t0][x + 12][y + 12][z + 10] - tau_xz[t0][x + 12][y + 12][z + 13]) + 1.22133638e-1F*(-tau_xx[t0][x + 12][y + 12][z + 12] + tau_xx[t0][x + 13][y + 12][z + 12] - tau_xy[t0][x + 12][y + 11][z + 12] + tau_xy[t0][x + 12][y + 12][z + 12] - tau_xz[t0][x + 12][y + 12][z + 11] + tau_xz[t0][x + 12][y + 12][z + 12]))*damp[x + 1][y + 1][z + 1] + damp[x + 1][y + 1][z + 1]*v_x[t0][x + 12][y + 12][z + 12];
              v_y[t1][x + 12][y + 12][z + 12] = 7.00999975204468e-1F*(irho[x + 12][y + 12][z + 12] + irho[x + 12][y + 13][z + 12])*(2.18478119e-6F*(tau_xy[t0][x + 6][y + 12][z + 12] - tau_xy[t0][x + 17][y + 12][z + 12] + tau_yy[t0][x + 12][y + 7][z + 12] - tau_yy[t0][x + 12][y + 18][z + 12] + tau_yz[t0][x + 12][y + 12][z + 6] - tau_yz[t0][x + 12][y + 12][z + 17]) + 3.59005404e-5F*(-tau_xy[t0][x + 7][y + 12][z + 12] + tau_xy[t0][x + 16][y + 12][z + 12] - tau_yy[t0][x + 12][y + 8][z + 12] + tau_yy[t0][x + 12][y + 17][z + 12] - tau_yz[t0][x + 12][y + 12][z + 7] + tau_yz[t0][x + 12][y + 12][z + 16]) + 2.96728956e-4F*(tau_xy[t0][x + 8][y + 12][z + 12] - tau_xy[t0][x + 15][y + 12][z + 12] + tau_yy[t0][x + 12][y + 9][z + 12] - tau_yy[t0][x + 12][y + 16][z + 12] + tau_yz[t0][x + 12][y + 12][z + 8] - tau_yz[t0][x + 12][y + 12][z + 15]) + 1.74476626e-3F*(-tau_xy[t0][x + 9][y + 12][z + 12] + tau_xy[t0][x + 14][y + 12][z + 12] - tau_yy[t0][x + 12][y + 10][z + 12] + tau_yy[t0][x + 12][y + 15][z + 12] - tau_yz[t0][x + 12][y + 12][z + 9] + tau_yz[t0][x + 12][y + 12][z + 14]) + 9.6931459e-3F*(tau_xy[t0][x + 10][y + 12][z + 12] - tau_xy[t0][x + 13][y + 12][z + 12] + tau_yy[t0][x + 12][y + 11][z + 12] - tau_yy[t0][x + 12][y + 14][z + 12] + tau_yz[t0][x + 12][y + 12][z + 10] - tau_yz[t0][x + 12][y + 12][z + 13]) + 1.22133638e-1F*(-tau_xy[t0][x + 11][y + 12][z + 12] + tau_xy[t0][x + 12][y + 12][z + 12] - tau_yy[t0][x + 12][y + 12][z + 12] + tau_yy[t0][x + 12][y + 13][z + 12] - tau_yz[t0][x + 12][y + 12][z + 11] + tau_yz[t0][x + 12][y + 12][z + 12]))*damp[x + 1][y + 1][z + 1] + damp[x + 1][y + 1][z + 1]*v_y[t0][x + 12][y + 12][z + 12];
              v_z[t1][x + 12][y + 12][z + 12] = 7.00999975204468e-1F*(irho[x + 12][y + 12][z + 12] + irho[x + 12][y + 12][z + 13])*(2.18478119e-6F*(tau_xz[t0][x + 6][y + 12][z + 12] - tau_xz[t0][x + 17][y + 12][z + 12] + tau_yz[t0][x + 12][y + 6][z + 12] - tau_yz[t0][x + 12][y + 17][z + 12] + tau_zz[t0][x + 12][y + 12][z + 7] - tau_zz[t0][x + 12][y + 12][z + 18]) + 3.59005404e-5F*(-tau_xz[t0][x + 7][y + 12][z + 12] + tau_xz[t0][x + 16][y + 12][z + 12] - tau_yz[t0][x + 12][y + 7][z + 12] + tau_yz[t0][x + 12][y + 16][z + 12] - tau_zz[t0][x + 12][y + 12][z + 8] + tau_zz[t0][x + 12][y + 12][z + 17]) + 2.96728956e-4F*(tau_xz[t0][x + 8][y + 12][z + 12] - tau_xz[t0][x + 15][y + 12][z + 12] + tau_yz[t0][x + 12][y + 8][z + 12] - tau_yz[t0][x + 12][y + 15][z + 12] + tau_zz[t0][x + 12][y + 12][z + 9] - tau_zz[t0][x + 12][y + 12][z + 16]) + 1.74476626e-3F*(-tau_xz[t0][x + 9][y + 12][z + 12] + tau_xz[t0][x + 14][y + 12][z + 12] - tau_yz[t0][x + 12][y + 9][z + 12] + tau_yz[t0][x + 12][y + 14][z + 12] - tau_zz[t0][x + 12][y + 12][z + 10] + tau_zz[t0][x + 12][y + 12][z + 15]) + 9.6931459e-3F*(tau_xz[t0][x + 10][y + 12][z + 12] - tau_xz[t0][x + 13][y + 12][z + 12] + tau_yz[t0][x + 12][y + 10][z + 12] - tau_yz[t0][x + 12][y + 13][z + 12] + tau_zz[t0][x + 12][y + 12][z + 11] - tau_zz[t0][x + 12][y + 12][z + 14]) + 1.22133638e-1F*(-tau_xz[t0][x + 11][y + 12][z + 12] + tau_xz[t0][x + 12][y + 12][z + 12] - tau_yz[t0][x + 12][y + 11][z + 12] + tau_yz[t0][x + 12][y + 12][z + 12] - tau_zz[t0][x + 12][y + 12][z + 12] + tau_zz[t0][x + 12][y + 12][z + 13]))*damp[x + 1][y + 1][z + 1] + damp[x + 1][y + 1][z + 1]*v_z[t0][x + 12][y + 12][z + 12];
            }
          }
        }
      }
    }
  }
}

void bf1(struct dataobj *restrict damp_vec, struct dataobj *restrict lam_vec, struct dataobj *restrict mu_vec, struct dataobj *restrict tau_xx_vec, struct dataobj *restrict tau_xy_vec, struct dataobj *restrict tau_xz_vec, struct dataobj *restrict tau_yy_vec, struct dataobj *restrict tau_yz_vec, struct dataobj *restrict tau_zz_vec, struct dataobj *restrict v_x_vec, struct dataobj *restrict v_y_vec, struct dataobj *restrict v_z_vec, const int t0, const int t1, const int x1_blk0_size, const int x_M, const int x_m, const int y1_blk0_size, const int y_M, const int y_m, const int z_M, const int z_m, const int nthreads)
{
  float (*restrict damp)[damp_vec->size[1]][damp_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[damp_vec->size[1]][damp_vec->size[2]]) damp_vec->data;
  float (*restrict lam)[lam_vec->size[1]][lam_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[lam_vec->size[1]][lam_vec->size[2]]) lam_vec->data;
  float (*restrict mu)[mu_vec->size[1]][mu_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[mu_vec->size[1]][mu_vec->size[2]]) mu_vec->data;
  float (*restrict tau_xx)[tau_xx_vec->size[1]][tau_xx_vec->size[2]][tau_xx_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[tau_xx_vec->size[1]][tau_xx_vec->size[2]][tau_xx_vec->size[3]]) tau_xx_vec->data;
  float (*restrict tau_xy)[tau_xy_vec->size[1]][tau_xy_vec->size[2]][tau_xy_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[tau_xy_vec->size[1]][tau_xy_vec->size[2]][tau_xy_vec->size[3]]) tau_xy_vec->data;
  float (*restrict tau_xz)[tau_xz_vec->size[1]][tau_xz_vec->size[2]][tau_xz_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[tau_xz_vec->size[1]][tau_xz_vec->size[2]][tau_xz_vec->size[3]]) tau_xz_vec->data;
  float (*restrict tau_yy)[tau_yy_vec->size[1]][tau_yy_vec->size[2]][tau_yy_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[tau_yy_vec->size[1]][tau_yy_vec->size[2]][tau_yy_vec->size[3]]) tau_yy_vec->data;
  float (*restrict tau_yz)[tau_yz_vec->size[1]][tau_yz_vec->size[2]][tau_yz_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[tau_yz_vec->size[1]][tau_yz_vec->size[2]][tau_yz_vec->size[3]]) tau_yz_vec->data;
  float (*restrict tau_zz)[tau_zz_vec->size[1]][tau_zz_vec->size[2]][tau_zz_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[tau_zz_vec->size[1]][tau_zz_vec->size[2]][tau_zz_vec->size[3]]) tau_zz_vec->data;
  float (*restrict v_x)[v_x_vec->size[1]][v_x_vec->size[2]][v_x_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[v_x_vec->size[1]][v_x_vec->size[2]][v_x_vec->size[3]]) v_x_vec->data;
  float (*restrict v_y)[v_y_vec->size[1]][v_y_vec->size[2]][v_y_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[v_y_vec->size[1]][v_y_vec->size[2]][v_y_vec->size[3]]) v_y_vec->data;
  float (*restrict v_z)[v_z_vec->size[1]][v_z_vec->size[2]][v_z_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[v_z_vec->size[1]][v_z_vec->size[2]][v_z_vec->size[3]]) v_z_vec->data;
  if (x1_blk0_size == 0)
  {
    return;
  }
  #pragma omp parallel num_threads(nthreads)
  {
    #pragma omp for collapse(1) schedule(dynamic,1)
    for (int x1_blk0 = x_m; x1_blk0 <= x_M; x1_blk0 += x1_blk0_size)
    {
      for (int y1_blk0 = y_m; y1_blk0 <= y_M; y1_blk0 += y1_blk0_size)
      {
        for (int x = x1_blk0; x <= x1_blk0 + x1_blk0_size - 1; x += 1)
        {
          for (int y = y1_blk0; y <= y1_blk0 + y1_blk0_size - 1; y += 1)
          {
            #pragma omp simd aligned(damp,lam,mu,tau_xx,tau_xy,tau_xz,tau_yy,tau_yz,tau_zz,v_x,v_y,v_z:32)
            for (int z = z_m; z <= z_M; z += 1)
            {
              float r70 = -v_z[t1][x + 12][y + 12][z + 12];
              float r69 = -v_y[t1][x + 12][y + 12][z + 12];
              float r68 = -v_x[t1][x + 12][y + 12][z + 12];
              float r67 = -v_z[t1][x + 12][y + 12][z + 11];
              float r66 = -v_y[t1][x + 12][y + 11][z + 12];
              float r65 = -v_x[t1][x + 11][y + 12][z + 12];
              float r64 = -v_z[t1][x + 12][y + 12][z + 9];
              float r63 = -v_y[t1][x + 12][y + 9][z + 12];
              float r62 = -v_x[t1][x + 9][y + 12][z + 12];
              float r61 = -v_z[t1][x + 12][y + 12][z + 13];
              float r60 = -v_y[t1][x + 12][y + 13][z + 12];
              float r59 = -v_x[t1][x + 13][y + 12][z + 12];
              float r58 = -v_z[t1][x + 12][y + 12][z + 15];
              float r57 = -v_y[t1][x + 12][y + 15][z + 12];
              float r56 = -v_x[t1][x + 15][y + 12][z + 12];
              float r55 = -v_z[t1][x + 12][y + 12][z + 17];
              float r54 = -v_y[t1][x + 12][y + 17][z + 12];
              float r53 = -v_x[t1][x + 17][y + 12][z + 12];
              float r52 = -v_z[t1][x + 12][y + 12][z + 7];
              float r51 = -v_y[t1][x + 12][y + 7][z + 12];
              float r50 = -v_x[t1][x + 7][y + 12][z + 12];
              float r49 = 1.402F*(3.59005404e-5F*(r50 + r51 + r52 + v_x[t1][x + 16][y + 12][z + 12] + v_y[t1][x + 12][y + 16][z + 12] + v_z[t1][x + 12][y + 12][z + 16]) + 2.18478119e-6F*(r53 + r54 + r55 + v_x[t1][x + 6][y + 12][z + 12] + v_y[t1][x + 12][y + 6][z + 12] + v_z[t1][x + 12][y + 12][z + 6]) + 2.96728956e-4F*(r56 + r57 + r58 + v_x[t1][x + 8][y + 12][z + 12] + v_y[t1][x + 12][y + 8][z + 12] + v_z[t1][x + 12][y + 12][z + 8]) + 9.6931459e-3F*(r59 + r60 + r61 + v_x[t1][x + 10][y + 12][z + 12] + v_y[t1][x + 12][y + 10][z + 12] + v_z[t1][x + 12][y + 12][z + 10]) + 1.74476626e-3F*(r62 + r63 + r64 + v_x[t1][x + 14][y + 12][z + 12] + v_y[t1][x + 12][y + 14][z + 12] + v_z[t1][x + 12][y + 12][z + 14]) + 1.22133638e-1F*(r65 + r66 + r67 + v_x[t1][x + 12][y + 12][z + 12] + v_y[t1][x + 12][y + 12][z + 12] + v_z[t1][x + 12][y + 12][z + 12]))*damp[x + 1][y + 1][z + 1]*lam[x + 12][y + 12][z + 12];
              tau_xx[t1][x + 12][y + 12][z + 12] = r49 + 2.804F*(3.59005404e-5F*(r50 + v_x[t1][x + 16][y + 12][z + 12]) + 2.18478119e-6F*(r53 + v_x[t1][x + 6][y + 12][z + 12]) + 2.96728956e-4F*(r56 + v_x[t1][x + 8][y + 12][z + 12]) + 9.6931459e-3F*(r59 + v_x[t1][x + 10][y + 12][z + 12]) + 1.74476626e-3F*(r62 + v_x[t1][x + 14][y + 12][z + 12]) + 1.22133638e-1F*(r65 + v_x[t1][x + 12][y + 12][z + 12]))*damp[x + 1][y + 1][z + 1]*mu[x + 12][y + 12][z + 12] + damp[x + 1][y + 1][z + 1]*tau_xx[t0][x + 12][y + 12][z + 12];
              tau_xy[t1][x + 12][y + 12][z + 12] = 3.50499987602234e-1F*(mu[x + 12][y + 12][z + 12] + mu[x + 12][y + 13][z + 12] + mu[x + 13][y + 12][z + 12] + mu[x + 13][y + 13][z + 12])*(1.22133638e-1F*(r68 + r69 + v_x[t1][x + 12][y + 13][z + 12] + v_y[t1][x + 13][y + 12][z + 12]) + 2.18478119e-6F*(v_x[t1][x + 12][y + 7][z + 12] - v_x[t1][x + 12][y + 18][z + 12] + v_y[t1][x + 7][y + 12][z + 12] - v_y[t1][x + 18][y + 12][z + 12]) + 3.59005404e-5F*(-v_x[t1][x + 12][y + 8][z + 12] + v_x[t1][x + 12][y + 17][z + 12] - v_y[t1][x + 8][y + 12][z + 12] + v_y[t1][x + 17][y + 12][z + 12]) + 2.96728956e-4F*(v_x[t1][x + 12][y + 9][z + 12] - v_x[t1][x + 12][y + 16][z + 12] + v_y[t1][x + 9][y + 12][z + 12] - v_y[t1][x + 16][y + 12][z + 12]) + 1.74476626e-3F*(-v_x[t1][x + 12][y + 10][z + 12] + v_x[t1][x + 12][y + 15][z + 12] - v_y[t1][x + 10][y + 12][z + 12] + v_y[t1][x + 15][y + 12][z + 12]) + 9.6931459e-3F*(v_x[t1][x + 12][y + 11][z + 12] - v_x[t1][x + 12][y + 14][z + 12] + v_y[t1][x + 11][y + 12][z + 12] - v_y[t1][x + 14][y + 12][z + 12]))*damp[x + 1][y + 1][z + 1] + damp[x + 1][y + 1][z + 1]*tau_xy[t0][x + 12][y + 12][z + 12];
              tau_xz[t1][x + 12][y + 12][z + 12] = 3.50499987602234e-1F*(mu[x + 12][y + 12][z + 12] + mu[x + 12][y + 12][z + 13] + mu[x + 13][y + 12][z + 12] + mu[x + 13][y + 12][z + 13])*(1.22133638e-1F*(r68 + r70 + v_x[t1][x + 12][y + 12][z + 13] + v_z[t1][x + 13][y + 12][z + 12]) + 2.18478119e-6F*(v_x[t1][x + 12][y + 12][z + 7] - v_x[t1][x + 12][y + 12][z + 18] + v_z[t1][x + 7][y + 12][z + 12] - v_z[t1][x + 18][y + 12][z + 12]) + 3.59005404e-5F*(-v_x[t1][x + 12][y + 12][z + 8] + v_x[t1][x + 12][y + 12][z + 17] - v_z[t1][x + 8][y + 12][z + 12] + v_z[t1][x + 17][y + 12][z + 12]) + 2.96728956e-4F*(v_x[t1][x + 12][y + 12][z + 9] - v_x[t1][x + 12][y + 12][z + 16] + v_z[t1][x + 9][y + 12][z + 12] - v_z[t1][x + 16][y + 12][z + 12]) + 1.74476626e-3F*(-v_x[t1][x + 12][y + 12][z + 10] + v_x[t1][x + 12][y + 12][z + 15] - v_z[t1][x + 10][y + 12][z + 12] + v_z[t1][x + 15][y + 12][z + 12]) + 9.6931459e-3F*(v_x[t1][x + 12][y + 12][z + 11] - v_x[t1][x + 12][y + 12][z + 14] + v_z[t1][x + 11][y + 12][z + 12] - v_z[t1][x + 14][y + 12][z + 12]))*damp[x + 1][y + 1][z + 1] + damp[x + 1][y + 1][z + 1]*tau_xz[t0][x + 12][y + 12][z + 12];
              tau_yy[t1][x + 12][y + 12][z + 12] = r49 + 2.804F*(3.59005404e-5F*(r51 + v_y[t1][x + 12][y + 16][z + 12]) + 2.18478119e-6F*(r54 + v_y[t1][x + 12][y + 6][z + 12]) + 2.96728956e-4F*(r57 + v_y[t1][x + 12][y + 8][z + 12]) + 9.6931459e-3F*(r60 + v_y[t1][x + 12][y + 10][z + 12]) + 1.74476626e-3F*(r63 + v_y[t1][x + 12][y + 14][z + 12]) + 1.22133638e-1F*(r66 + v_y[t1][x + 12][y + 12][z + 12]))*damp[x + 1][y + 1][z + 1]*mu[x + 12][y + 12][z + 12] + damp[x + 1][y + 1][z + 1]*tau_yy[t0][x + 12][y + 12][z + 12];
              tau_yz[t1][x + 12][y + 12][z + 12] = 3.50499987602234e-1F*(mu[x + 12][y + 12][z + 12] + mu[x + 12][y + 12][z + 13] + mu[x + 12][y + 13][z + 12] + mu[x + 12][y + 13][z + 13])*(1.22133638e-1F*(r69 + r70 + v_y[t1][x + 12][y + 12][z + 13] + v_z[t1][x + 12][y + 13][z + 12]) + 2.18478119e-6F*(v_y[t1][x + 12][y + 12][z + 7] - v_y[t1][x + 12][y + 12][z + 18] + v_z[t1][x + 12][y + 7][z + 12] - v_z[t1][x + 12][y + 18][z + 12]) + 3.59005404e-5F*(-v_y[t1][x + 12][y + 12][z + 8] + v_y[t1][x + 12][y + 12][z + 17] - v_z[t1][x + 12][y + 8][z + 12] + v_z[t1][x + 12][y + 17][z + 12]) + 2.96728956e-4F*(v_y[t1][x + 12][y + 12][z + 9] - v_y[t1][x + 12][y + 12][z + 16] + v_z[t1][x + 12][y + 9][z + 12] - v_z[t1][x + 12][y + 16][z + 12]) + 1.74476626e-3F*(-v_y[t1][x + 12][y + 12][z + 10] + v_y[t1][x + 12][y + 12][z + 15] - v_z[t1][x + 12][y + 10][z + 12] + v_z[t1][x + 12][y + 15][z + 12]) + 9.6931459e-3F*(v_y[t1][x + 12][y + 12][z + 11] - v_y[t1][x + 12][y + 12][z + 14] + v_z[t1][x + 12][y + 11][z + 12] - v_z[t1][x + 12][y + 14][z + 12]))*damp[x + 1][y + 1][z + 1] + damp[x + 1][y + 1][z + 1]*tau_yz[t0][x + 12][y + 12][z + 12];
              tau_zz[t1][x + 12][y + 12][z + 12] = r49 + 2.804F*(3.59005404e-5F*(r52 + v_z[t1][x + 12][y + 12][z + 16]) + 2.18478119e-6F*(r55 + v_z[t1][x + 12][y + 12][z + 6]) + 2.96728956e-4F*(r58 + v_z[t1][x + 12][y + 12][z + 8]) + 9.6931459e-3F*(r61 + v_z[t1][x + 12][y + 12][z + 10]) + 1.74476626e-3F*(r64 + v_z[t1][x + 12][y + 12][z + 14]) + 1.22133638e-1F*(r67 + v_z[t1][x + 12][y + 12][z + 12]))*damp[x + 1][y + 1][z + 1]*mu[x + 12][y + 12][z + 12] + damp[x + 1][y + 1][z + 1]*tau_zz[t0][x + 12][y + 12][z + 12];
            }
          }
        }
      }
    }
  }
}
