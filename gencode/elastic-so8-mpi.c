#define _POSIX_C_SOURCE 200809L
#include "stdlib.h"
#include "math.h"
#include "sys/time.h"
#include "xmmintrin.h"
#include "pmmintrin.h"
#include "mpi.h"
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

struct neighborhood
{
  int lll, llc, llr, lcl, lcc, lcr, lrl, lrc, lrr;
  int cll, clc, clr, ccl, ccc, ccr, crl, crc, crr;
  int rll, rlc, rlr, rcl, rcc, rcr, rrl, rrc, rrr;
} ;

struct profiler
{
  double section0;
  double section1;
  double section2;
  double section3;
} ;

void sendrecv_txyz(struct dataobj *restrict a_vec, const int buf_x_size, const int buf_y_size, const int buf_z_size, int ogtime, int ogx, int ogy, int ogz, int ostime, int osx, int osy, int osz, int fromrank, int torank, MPI_Comm comm, const int nthreads);
void gather_txyz(float *restrict buf_vec, const int buf_x_size, const int buf_y_size, const int buf_z_size, struct dataobj *restrict a_vec, int otime, int ox, int oy, int oz, const int nthreads);
void scatter_txyz(float *restrict buf_vec, const int buf_x_size, const int buf_y_size, const int buf_z_size, struct dataobj *restrict a_vec, int otime, int ox, int oy, int oz, const int nthreads);
void haloupdate7(struct dataobj *restrict a_vec, MPI_Comm comm, struct neighborhood * nb, int otime, const int nthreads);
void haloupdate0(struct dataobj *restrict a_vec, MPI_Comm comm, struct neighborhood * nb, int otime, const int nthreads);
void haloupdate1(struct dataobj *restrict a_vec, MPI_Comm comm, struct neighborhood * nb, int otime, const int nthreads);
void haloupdate2(struct dataobj *restrict a_vec, MPI_Comm comm, struct neighborhood * nb, int otime, const int nthreads);
void haloupdate3(struct dataobj *restrict a_vec, MPI_Comm comm, struct neighborhood * nb, int otime, const int nthreads);
void haloupdate4(struct dataobj *restrict a_vec, MPI_Comm comm, struct neighborhood * nb, int otime, const int nthreads);
void haloupdate5(struct dataobj *restrict a_vec, MPI_Comm comm, struct neighborhood * nb, int otime, const int nthreads);
void haloupdate6(struct dataobj *restrict a_vec, MPI_Comm comm, struct neighborhood * nb, int otime, const int nthreads);
void bf0(struct dataobj *restrict damp_vec, struct dataobj *restrict irho_vec, struct dataobj *restrict tau_xx_vec, struct dataobj *restrict tau_xy_vec, struct dataobj *restrict tau_xz_vec, struct dataobj *restrict tau_yy_vec, struct dataobj *restrict tau_yz_vec, struct dataobj *restrict tau_zz_vec, struct dataobj *restrict v_x_vec, struct dataobj *restrict v_y_vec, struct dataobj *restrict v_z_vec, const int t0, const int t1, const int x0_blk0_size, const int x_M, const int x_m, const int y0_blk0_size, const int y_M, const int y_m, const int z_M, const int z_m, const int nthreads);
void bf1(struct dataobj *restrict damp_vec, struct dataobj *restrict lam_vec, struct dataobj *restrict mu_vec, struct dataobj *restrict tau_xx_vec, struct dataobj *restrict tau_xy_vec, struct dataobj *restrict tau_xz_vec, struct dataobj *restrict tau_yy_vec, struct dataobj *restrict tau_yz_vec, struct dataobj *restrict tau_zz_vec, struct dataobj *restrict v_x_vec, struct dataobj *restrict v_y_vec, struct dataobj *restrict v_z_vec, const int t0, const int t1, const int x1_blk0_size, const int x_M, const int x_m, const int y1_blk0_size, const int y_M, const int y_m, const int z_M, const int z_m, const int nthreads);

int ForwardElastic(struct dataobj *restrict damp_vec, const float dt, struct dataobj *restrict irho_vec, struct dataobj *restrict lam_vec, struct dataobj *restrict mu_vec, const float o_x, const float o_y, const float o_z, struct dataobj *restrict rec1_vec, struct dataobj *restrict rec1_coords_vec, struct dataobj *restrict rec2_vec, struct dataobj *restrict rec2_coords_vec, struct dataobj *restrict src_vec, struct dataobj *restrict src_coords_vec, struct dataobj *restrict tau_xx_vec, struct dataobj *restrict tau_xy_vec, struct dataobj *restrict tau_xz_vec, struct dataobj *restrict tau_yy_vec, struct dataobj *restrict tau_yz_vec, struct dataobj *restrict tau_zz_vec, struct dataobj *restrict v_x_vec, struct dataobj *restrict v_y_vec, struct dataobj *restrict v_z_vec, const int x_M, const int x_m, const int y_M, const int y_m, const int z_M, const int z_m, const int p_rec1_M, const int p_rec1_m, const int p_rec2_M, const int p_rec2_m, const int p_src_M, const int p_src_m, const int time_M, const int time_m, struct profiler * timers, const int x0_blk0_size, const int x1_blk0_size, const int y0_blk0_size, const int y1_blk0_size, MPI_Comm comm, struct neighborhood * nb, const int nthreads, const int nthreads_nonaffine)
{
  float (*restrict damp)[damp_vec->size[1]][damp_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[damp_vec->size[1]][damp_vec->size[2]]) damp_vec->data;
  float (*restrict irho)[irho_vec->size[1]][irho_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[irho_vec->size[1]][irho_vec->size[2]]) irho_vec->data;
  float (*restrict lam)[lam_vec->size[1]][lam_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[lam_vec->size[1]][lam_vec->size[2]]) lam_vec->data;
  float (*restrict mu)[mu_vec->size[1]][mu_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[mu_vec->size[1]][mu_vec->size[2]]) mu_vec->data;
  float (*restrict rec1)[rec1_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[rec1_vec->size[1]]) rec1_vec->data;
  float (*restrict rec1_coords)[rec1_coords_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[rec1_coords_vec->size[1]]) rec1_coords_vec->data;
  float (*restrict rec2)[rec2_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[rec2_vec->size[1]]) rec2_vec->data;
  float (*restrict rec2_coords)[rec2_coords_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[rec2_coords_vec->size[1]]) rec2_coords_vec->data;
  float (*restrict src)[src_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[src_vec->size[1]]) src_vec->data;
  float (*restrict src_coords)[src_coords_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[src_coords_vec->size[1]]) src_coords_vec->data;
  float (*restrict tau_xx)[tau_xx_vec->size[1]][tau_xx_vec->size[2]][tau_xx_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[tau_xx_vec->size[1]][tau_xx_vec->size[2]][tau_xx_vec->size[3]]) tau_xx_vec->data;
  float (*restrict tau_xy)[tau_xy_vec->size[1]][tau_xy_vec->size[2]][tau_xy_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[tau_xy_vec->size[1]][tau_xy_vec->size[2]][tau_xy_vec->size[3]]) tau_xy_vec->data;
  float (*restrict tau_xz)[tau_xz_vec->size[1]][tau_xz_vec->size[2]][tau_xz_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[tau_xz_vec->size[1]][tau_xz_vec->size[2]][tau_xz_vec->size[3]]) tau_xz_vec->data;
  float (*restrict tau_yy)[tau_yy_vec->size[1]][tau_yy_vec->size[2]][tau_yy_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[tau_yy_vec->size[1]][tau_yy_vec->size[2]][tau_yy_vec->size[3]]) tau_yy_vec->data;
  float (*restrict tau_yz)[tau_yz_vec->size[1]][tau_yz_vec->size[2]][tau_yz_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[tau_yz_vec->size[1]][tau_yz_vec->size[2]][tau_yz_vec->size[3]]) tau_yz_vec->data;
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
    haloupdate0(tau_xx_vec,comm,nb,t0,nthreads);
    haloupdate1(tau_xy_vec,comm,nb,t0,nthreads);
    haloupdate2(tau_xz_vec,comm,nb,t0,nthreads);
    haloupdate3(tau_yy_vec,comm,nb,t0,nthreads);
    haloupdate4(tau_yz_vec,comm,nb,t0,nthreads);
    haloupdate5(tau_zz_vec,comm,nb,t0,nthreads);
    bf0(damp_vec,irho_vec,tau_xx_vec,tau_xy_vec,tau_xz_vec,tau_yy_vec,tau_yz_vec,tau_zz_vec,v_x_vec,v_y_vec,v_z_vec,t0,t1,x0_blk0_size,x_M - (x_M - x_m + 1)%(x0_blk0_size),x_m,y0_blk0_size,y_M - (y_M - y_m + 1)%(y0_blk0_size),y_m,z_M,z_m,nthreads);
    bf0(damp_vec,irho_vec,tau_xx_vec,tau_xy_vec,tau_xz_vec,tau_yy_vec,tau_yz_vec,tau_zz_vec,v_x_vec,v_y_vec,v_z_vec,t0,t1,x0_blk0_size,x_M - (x_M - x_m + 1)%(x0_blk0_size),x_m,(y_M - y_m + 1)%(y0_blk0_size),y_M,y_M - (y_M - y_m + 1)%(y0_blk0_size) + 1,z_M,z_m,nthreads);
    bf0(damp_vec,irho_vec,tau_xx_vec,tau_xy_vec,tau_xz_vec,tau_yy_vec,tau_yz_vec,tau_zz_vec,v_x_vec,v_y_vec,v_z_vec,t0,t1,(x_M - x_m + 1)%(x0_blk0_size),x_M,x_M - (x_M - x_m + 1)%(x0_blk0_size) + 1,y0_blk0_size,y_M - (y_M - y_m + 1)%(y0_blk0_size),y_m,z_M,z_m,nthreads);
    bf0(damp_vec,irho_vec,tau_xx_vec,tau_xy_vec,tau_xz_vec,tau_yy_vec,tau_yz_vec,tau_zz_vec,v_x_vec,v_y_vec,v_z_vec,t0,t1,(x_M - x_m + 1)%(x0_blk0_size),x_M,x_M - (x_M - x_m + 1)%(x0_blk0_size) + 1,(y_M - y_m + 1)%(y0_blk0_size),y_M,y_M - (y_M - y_m + 1)%(y0_blk0_size) + 1,z_M,z_m,nthreads);
    haloupdate6(v_x_vec,comm,nb,t1,nthreads);
    haloupdate6(v_y_vec,comm,nb,t1,nthreads);
    haloupdate6(v_z_vec,comm,nb,t1,nthreads);
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
          tau_xx[t1][ii_src_0 + 8][ii_src_1 + 8][ii_src_2 + 8] += r0;
        }
        if (ii_src_0 >= x_m - 1 && ii_src_1 >= y_m - 1 && ii_src_3 >= z_m - 1 && ii_src_0 <= x_M + 1 && ii_src_1 <= y_M + 1 && ii_src_3 <= z_M + 1)
        {
          float r1 = dt*(1.0e-3F*px*py*pz - 1.0e-2F*px*pz - 1.0e-2F*py*pz + 1.0e-1F*pz)*src[time][p_src];
          #pragma omp atomic update
          tau_xx[t1][ii_src_0 + 8][ii_src_1 + 8][ii_src_3 + 8] += r1;
        }
        if (ii_src_0 >= x_m - 1 && ii_src_2 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_0 <= x_M + 1 && ii_src_2 <= z_M + 1 && ii_src_4 <= y_M + 1)
        {
          float r2 = dt*(1.0e-3F*px*py*pz - 1.0e-2F*px*py - 1.0e-2F*py*pz + 1.0e-1F*py)*src[time][p_src];
          #pragma omp atomic update
          tau_xx[t1][ii_src_0 + 8][ii_src_4 + 8][ii_src_2 + 8] += r2;
        }
        if (ii_src_0 >= x_m - 1 && ii_src_3 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_0 <= x_M + 1 && ii_src_3 <= z_M + 1 && ii_src_4 <= y_M + 1)
        {
          float r3 = dt*(-1.0e-3F*px*py*pz + 1.0e-2F*py*pz)*src[time][p_src];
          #pragma omp atomic update
          tau_xx[t1][ii_src_0 + 8][ii_src_4 + 8][ii_src_3 + 8] += r3;
        }
        if (ii_src_1 >= y_m - 1 && ii_src_2 >= z_m - 1 && ii_src_5 >= x_m - 1 && ii_src_1 <= y_M + 1 && ii_src_2 <= z_M + 1 && ii_src_5 <= x_M + 1)
        {
          float r4 = dt*(1.0e-3F*px*py*pz - 1.0e-2F*px*py - 1.0e-2F*px*pz + 1.0e-1F*px)*src[time][p_src];
          #pragma omp atomic update
          tau_xx[t1][ii_src_5 + 8][ii_src_1 + 8][ii_src_2 + 8] += r4;
        }
        if (ii_src_1 >= y_m - 1 && ii_src_3 >= z_m - 1 && ii_src_5 >= x_m - 1 && ii_src_1 <= y_M + 1 && ii_src_3 <= z_M + 1 && ii_src_5 <= x_M + 1)
        {
          float r5 = dt*(-1.0e-3F*px*py*pz + 1.0e-2F*px*pz)*src[time][p_src];
          #pragma omp atomic update
          tau_xx[t1][ii_src_5 + 8][ii_src_1 + 8][ii_src_3 + 8] += r5;
        }
        if (ii_src_2 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_5 >= x_m - 1 && ii_src_2 <= z_M + 1 && ii_src_4 <= y_M + 1 && ii_src_5 <= x_M + 1)
        {
          float r6 = dt*(-1.0e-3F*px*py*pz + 1.0e-2F*px*py)*src[time][p_src];
          #pragma omp atomic update
          tau_xx[t1][ii_src_5 + 8][ii_src_4 + 8][ii_src_2 + 8] += r6;
        }
        if (ii_src_3 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_5 >= x_m - 1 && ii_src_3 <= z_M + 1 && ii_src_4 <= y_M + 1 && ii_src_5 <= x_M + 1)
        {
          float r7 = 1.0e-3F*px*py*pz*dt*src[time][p_src];
          #pragma omp atomic update
          tau_xx[t1][ii_src_5 + 8][ii_src_4 + 8][ii_src_3 + 8] += r7;
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
          tau_zz[t1][ii_src_0 + 8][ii_src_1 + 8][ii_src_2 + 8] += r8;
        }
        if (ii_src_0 >= x_m - 1 && ii_src_1 >= y_m - 1 && ii_src_3 >= z_m - 1 && ii_src_0 <= x_M + 1 && ii_src_1 <= y_M + 1 && ii_src_3 <= z_M + 1)
        {
          float r9 = dt*(1.0e-3F*px*py*pz - 1.0e-2F*px*pz - 1.0e-2F*py*pz + 1.0e-1F*pz)*src[time][p_src];
          #pragma omp atomic update
          tau_zz[t1][ii_src_0 + 8][ii_src_1 + 8][ii_src_3 + 8] += r9;
        }
        if (ii_src_0 >= x_m - 1 && ii_src_2 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_0 <= x_M + 1 && ii_src_2 <= z_M + 1 && ii_src_4 <= y_M + 1)
        {
          float r10 = dt*(1.0e-3F*px*py*pz - 1.0e-2F*px*py - 1.0e-2F*py*pz + 1.0e-1F*py)*src[time][p_src];
          #pragma omp atomic update
          tau_zz[t1][ii_src_0 + 8][ii_src_4 + 8][ii_src_2 + 8] += r10;
        }
        if (ii_src_0 >= x_m - 1 && ii_src_3 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_0 <= x_M + 1 && ii_src_3 <= z_M + 1 && ii_src_4 <= y_M + 1)
        {
          float r11 = dt*(-1.0e-3F*px*py*pz + 1.0e-2F*py*pz)*src[time][p_src];
          #pragma omp atomic update
          tau_zz[t1][ii_src_0 + 8][ii_src_4 + 8][ii_src_3 + 8] += r11;
        }
        if (ii_src_1 >= y_m - 1 && ii_src_2 >= z_m - 1 && ii_src_5 >= x_m - 1 && ii_src_1 <= y_M + 1 && ii_src_2 <= z_M + 1 && ii_src_5 <= x_M + 1)
        {
          float r12 = dt*(1.0e-3F*px*py*pz - 1.0e-2F*px*py - 1.0e-2F*px*pz + 1.0e-1F*px)*src[time][p_src];
          #pragma omp atomic update
          tau_zz[t1][ii_src_5 + 8][ii_src_1 + 8][ii_src_2 + 8] += r12;
        }
        if (ii_src_1 >= y_m - 1 && ii_src_3 >= z_m - 1 && ii_src_5 >= x_m - 1 && ii_src_1 <= y_M + 1 && ii_src_3 <= z_M + 1 && ii_src_5 <= x_M + 1)
        {
          float r13 = dt*(-1.0e-3F*px*py*pz + 1.0e-2F*px*pz)*src[time][p_src];
          #pragma omp atomic update
          tau_zz[t1][ii_src_5 + 8][ii_src_1 + 8][ii_src_3 + 8] += r13;
        }
        if (ii_src_2 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_5 >= x_m - 1 && ii_src_2 <= z_M + 1 && ii_src_4 <= y_M + 1 && ii_src_5 <= x_M + 1)
        {
          float r14 = dt*(-1.0e-3F*px*py*pz + 1.0e-2F*px*py)*src[time][p_src];
          #pragma omp atomic update
          tau_zz[t1][ii_src_5 + 8][ii_src_4 + 8][ii_src_2 + 8] += r14;
        }
        if (ii_src_3 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_5 >= x_m - 1 && ii_src_3 <= z_M + 1 && ii_src_4 <= y_M + 1 && ii_src_5 <= x_M + 1)
        {
          float r15 = 1.0e-3F*px*py*pz*dt*src[time][p_src];
          #pragma omp atomic update
          tau_zz[t1][ii_src_5 + 8][ii_src_4 + 8][ii_src_3 + 8] += r15;
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
          tau_yy[t1][ii_src_0 + 8][ii_src_1 + 8][ii_src_2 + 8] += r16;
        }
        if (ii_src_0 >= x_m - 1 && ii_src_1 >= y_m - 1 && ii_src_3 >= z_m - 1 && ii_src_0 <= x_M + 1 && ii_src_1 <= y_M + 1 && ii_src_3 <= z_M + 1)
        {
          float r17 = dt*(1.0e-3F*px*py*pz - 1.0e-2F*px*pz - 1.0e-2F*py*pz + 1.0e-1F*pz)*src[time][p_src];
          #pragma omp atomic update
          tau_yy[t1][ii_src_0 + 8][ii_src_1 + 8][ii_src_3 + 8] += r17;
        }
        if (ii_src_0 >= x_m - 1 && ii_src_2 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_0 <= x_M + 1 && ii_src_2 <= z_M + 1 && ii_src_4 <= y_M + 1)
        {
          float r18 = dt*(1.0e-3F*px*py*pz - 1.0e-2F*px*py - 1.0e-2F*py*pz + 1.0e-1F*py)*src[time][p_src];
          #pragma omp atomic update
          tau_yy[t1][ii_src_0 + 8][ii_src_4 + 8][ii_src_2 + 8] += r18;
        }
        if (ii_src_0 >= x_m - 1 && ii_src_3 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_0 <= x_M + 1 && ii_src_3 <= z_M + 1 && ii_src_4 <= y_M + 1)
        {
          float r19 = dt*(-1.0e-3F*px*py*pz + 1.0e-2F*py*pz)*src[time][p_src];
          #pragma omp atomic update
          tau_yy[t1][ii_src_0 + 8][ii_src_4 + 8][ii_src_3 + 8] += r19;
        }
        if (ii_src_1 >= y_m - 1 && ii_src_2 >= z_m - 1 && ii_src_5 >= x_m - 1 && ii_src_1 <= y_M + 1 && ii_src_2 <= z_M + 1 && ii_src_5 <= x_M + 1)
        {
          float r20 = dt*(1.0e-3F*px*py*pz - 1.0e-2F*px*py - 1.0e-2F*px*pz + 1.0e-1F*px)*src[time][p_src];
          #pragma omp atomic update
          tau_yy[t1][ii_src_5 + 8][ii_src_1 + 8][ii_src_2 + 8] += r20;
        }
        if (ii_src_1 >= y_m - 1 && ii_src_3 >= z_m - 1 && ii_src_5 >= x_m - 1 && ii_src_1 <= y_M + 1 && ii_src_3 <= z_M + 1 && ii_src_5 <= x_M + 1)
        {
          float r21 = dt*(-1.0e-3F*px*py*pz + 1.0e-2F*px*pz)*src[time][p_src];
          #pragma omp atomic update
          tau_yy[t1][ii_src_5 + 8][ii_src_1 + 8][ii_src_3 + 8] += r21;
        }
        if (ii_src_2 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_5 >= x_m - 1 && ii_src_2 <= z_M + 1 && ii_src_4 <= y_M + 1 && ii_src_5 <= x_M + 1)
        {
          float r22 = dt*(-1.0e-3F*px*py*pz + 1.0e-2F*px*py)*src[time][p_src];
          #pragma omp atomic update
          tau_yy[t1][ii_src_5 + 8][ii_src_4 + 8][ii_src_2 + 8] += r22;
        }
        if (ii_src_3 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_5 >= x_m - 1 && ii_src_3 <= z_M + 1 && ii_src_4 <= y_M + 1 && ii_src_5 <= x_M + 1)
        {
          float r23 = 1.0e-3F*px*py*pz*dt*src[time][p_src];
          #pragma omp atomic update
          tau_yy[t1][ii_src_5 + 8][ii_src_4 + 8][ii_src_3 + 8] += r23;
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
          sum += (-1.0e-3F*px*py*pz + 1.0e-2F*px*py + 1.0e-2F*px*pz - 1.0e-1F*px + 1.0e-2F*py*pz - 1.0e-1F*py - 1.0e-1F*pz + 1)*tau_zz[t0][ii_rec1_0 + 8][ii_rec1_1 + 8][ii_rec1_2 + 8];
        }
        if (ii_rec1_0 >= x_m - 1 && ii_rec1_1 >= y_m - 1 && ii_rec1_3 >= z_m - 1 && ii_rec1_0 <= x_M + 1 && ii_rec1_1 <= y_M + 1 && ii_rec1_3 <= z_M + 1)
        {
          sum += (1.0e-3F*px*py*pz - 1.0e-2F*px*pz - 1.0e-2F*py*pz + 1.0e-1F*pz)*tau_zz[t0][ii_rec1_0 + 8][ii_rec1_1 + 8][ii_rec1_3 + 8];
        }
        if (ii_rec1_0 >= x_m - 1 && ii_rec1_2 >= z_m - 1 && ii_rec1_4 >= y_m - 1 && ii_rec1_0 <= x_M + 1 && ii_rec1_2 <= z_M + 1 && ii_rec1_4 <= y_M + 1)
        {
          sum += (1.0e-3F*px*py*pz - 1.0e-2F*px*py - 1.0e-2F*py*pz + 1.0e-1F*py)*tau_zz[t0][ii_rec1_0 + 8][ii_rec1_4 + 8][ii_rec1_2 + 8];
        }
        if (ii_rec1_0 >= x_m - 1 && ii_rec1_3 >= z_m - 1 && ii_rec1_4 >= y_m - 1 && ii_rec1_0 <= x_M + 1 && ii_rec1_3 <= z_M + 1 && ii_rec1_4 <= y_M + 1)
        {
          sum += (-1.0e-3F*px*py*pz + 1.0e-2F*py*pz)*tau_zz[t0][ii_rec1_0 + 8][ii_rec1_4 + 8][ii_rec1_3 + 8];
        }
        if (ii_rec1_1 >= y_m - 1 && ii_rec1_2 >= z_m - 1 && ii_rec1_5 >= x_m - 1 && ii_rec1_1 <= y_M + 1 && ii_rec1_2 <= z_M + 1 && ii_rec1_5 <= x_M + 1)
        {
          sum += (1.0e-3F*px*py*pz - 1.0e-2F*px*py - 1.0e-2F*px*pz + 1.0e-1F*px)*tau_zz[t0][ii_rec1_5 + 8][ii_rec1_1 + 8][ii_rec1_2 + 8];
        }
        if (ii_rec1_1 >= y_m - 1 && ii_rec1_3 >= z_m - 1 && ii_rec1_5 >= x_m - 1 && ii_rec1_1 <= y_M + 1 && ii_rec1_3 <= z_M + 1 && ii_rec1_5 <= x_M + 1)
        {
          sum += (-1.0e-3F*px*py*pz + 1.0e-2F*px*pz)*tau_zz[t0][ii_rec1_5 + 8][ii_rec1_1 + 8][ii_rec1_3 + 8];
        }
        if (ii_rec1_2 >= z_m - 1 && ii_rec1_4 >= y_m - 1 && ii_rec1_5 >= x_m - 1 && ii_rec1_2 <= z_M + 1 && ii_rec1_4 <= y_M + 1 && ii_rec1_5 <= x_M + 1)
        {
          sum += (-1.0e-3F*px*py*pz + 1.0e-2F*px*py)*tau_zz[t0][ii_rec1_5 + 8][ii_rec1_4 + 8][ii_rec1_2 + 8];
        }
        if (ii_rec1_3 >= z_m - 1 && ii_rec1_4 >= y_m - 1 && ii_rec1_5 >= x_m - 1 && ii_rec1_3 <= z_M + 1 && ii_rec1_4 <= y_M + 1 && ii_rec1_5 <= x_M + 1)
        {
          sum += 1.0e-3F*px*py*pz*tau_zz[t0][ii_rec1_5 + 8][ii_rec1_4 + 8][ii_rec1_3 + 8];
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
    haloupdate7(v_x_vec,comm,nb,t0,nthreads);
    haloupdate7(v_y_vec,comm,nb,t0,nthreads);
    haloupdate7(v_z_vec,comm,nb,t0,nthreads);
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
          sum += (-1.0e-3F*px*py*pz + 1.0e-2F*px*py + 1.0e-2F*px*pz - 1.0e-1F*px + 1.0e-2F*py*pz - 1.0e-1F*py - 1.0e-1F*pz + 1)*(3.57142862e-4F*v_x[t0][ii_rec2_0 + 4][ii_rec2_1 + 8][ii_rec2_2 + 8] - 3.80952387e-3F*v_x[t0][ii_rec2_0 + 5][ii_rec2_1 + 8][ii_rec2_2 + 8] + 2.00000003e-2F*v_x[t0][ii_rec2_0 + 6][ii_rec2_1 + 8][ii_rec2_2 + 8] - 8.00000012e-2F*v_x[t0][ii_rec2_0 + 7][ii_rec2_1 + 8][ii_rec2_2 + 8] + 8.00000012e-2F*v_x[t0][ii_rec2_0 + 9][ii_rec2_1 + 8][ii_rec2_2 + 8] - 2.00000003e-2F*v_x[t0][ii_rec2_0 + 10][ii_rec2_1 + 8][ii_rec2_2 + 8] + 3.80952387e-3F*v_x[t0][ii_rec2_0 + 11][ii_rec2_1 + 8][ii_rec2_2 + 8] - 3.57142862e-4F*v_x[t0][ii_rec2_0 + 12][ii_rec2_1 + 8][ii_rec2_2 + 8] + 3.57142862e-4F*v_y[t0][ii_rec2_0 + 8][ii_rec2_1 + 4][ii_rec2_2 + 8] - 3.80952387e-3F*v_y[t0][ii_rec2_0 + 8][ii_rec2_1 + 5][ii_rec2_2 + 8] + 2.00000003e-2F*v_y[t0][ii_rec2_0 + 8][ii_rec2_1 + 6][ii_rec2_2 + 8] - 8.00000012e-2F*v_y[t0][ii_rec2_0 + 8][ii_rec2_1 + 7][ii_rec2_2 + 8] + 8.00000012e-2F*v_y[t0][ii_rec2_0 + 8][ii_rec2_1 + 9][ii_rec2_2 + 8] - 2.00000003e-2F*v_y[t0][ii_rec2_0 + 8][ii_rec2_1 + 10][ii_rec2_2 + 8] + 3.80952387e-3F*v_y[t0][ii_rec2_0 + 8][ii_rec2_1 + 11][ii_rec2_2 + 8] - 3.57142862e-4F*v_y[t0][ii_rec2_0 + 8][ii_rec2_1 + 12][ii_rec2_2 + 8] + 3.57142862e-4F*v_z[t0][ii_rec2_0 + 8][ii_rec2_1 + 8][ii_rec2_2 + 4] - 3.80952387e-3F*v_z[t0][ii_rec2_0 + 8][ii_rec2_1 + 8][ii_rec2_2 + 5] + 2.00000003e-2F*v_z[t0][ii_rec2_0 + 8][ii_rec2_1 + 8][ii_rec2_2 + 6] - 8.00000012e-2F*v_z[t0][ii_rec2_0 + 8][ii_rec2_1 + 8][ii_rec2_2 + 7] + 8.00000012e-2F*v_z[t0][ii_rec2_0 + 8][ii_rec2_1 + 8][ii_rec2_2 + 9] - 2.00000003e-2F*v_z[t0][ii_rec2_0 + 8][ii_rec2_1 + 8][ii_rec2_2 + 10] + 3.80952387e-3F*v_z[t0][ii_rec2_0 + 8][ii_rec2_1 + 8][ii_rec2_2 + 11] - 3.57142862e-4F*v_z[t0][ii_rec2_0 + 8][ii_rec2_1 + 8][ii_rec2_2 + 12]);
        }
        if (ii_rec2_0 >= x_m - 1 && ii_rec2_1 >= y_m - 1 && ii_rec2_3 >= z_m - 1 && ii_rec2_0 <= x_M + 1 && ii_rec2_1 <= y_M + 1 && ii_rec2_3 <= z_M + 1)
        {
          sum += (1.0e-3F*px*py*pz - 1.0e-2F*px*pz - 1.0e-2F*py*pz + 1.0e-1F*pz)*(3.57142862e-4F*v_x[t0][ii_rec2_0 + 4][ii_rec2_1 + 8][ii_rec2_3 + 8] - 3.80952387e-3F*v_x[t0][ii_rec2_0 + 5][ii_rec2_1 + 8][ii_rec2_3 + 8] + 2.00000003e-2F*v_x[t0][ii_rec2_0 + 6][ii_rec2_1 + 8][ii_rec2_3 + 8] - 8.00000012e-2F*v_x[t0][ii_rec2_0 + 7][ii_rec2_1 + 8][ii_rec2_3 + 8] + 8.00000012e-2F*v_x[t0][ii_rec2_0 + 9][ii_rec2_1 + 8][ii_rec2_3 + 8] - 2.00000003e-2F*v_x[t0][ii_rec2_0 + 10][ii_rec2_1 + 8][ii_rec2_3 + 8] + 3.80952387e-3F*v_x[t0][ii_rec2_0 + 11][ii_rec2_1 + 8][ii_rec2_3 + 8] - 3.57142862e-4F*v_x[t0][ii_rec2_0 + 12][ii_rec2_1 + 8][ii_rec2_3 + 8] + 3.57142862e-4F*v_y[t0][ii_rec2_0 + 8][ii_rec2_1 + 4][ii_rec2_3 + 8] - 3.80952387e-3F*v_y[t0][ii_rec2_0 + 8][ii_rec2_1 + 5][ii_rec2_3 + 8] + 2.00000003e-2F*v_y[t0][ii_rec2_0 + 8][ii_rec2_1 + 6][ii_rec2_3 + 8] - 8.00000012e-2F*v_y[t0][ii_rec2_0 + 8][ii_rec2_1 + 7][ii_rec2_3 + 8] + 8.00000012e-2F*v_y[t0][ii_rec2_0 + 8][ii_rec2_1 + 9][ii_rec2_3 + 8] - 2.00000003e-2F*v_y[t0][ii_rec2_0 + 8][ii_rec2_1 + 10][ii_rec2_3 + 8] + 3.80952387e-3F*v_y[t0][ii_rec2_0 + 8][ii_rec2_1 + 11][ii_rec2_3 + 8] - 3.57142862e-4F*v_y[t0][ii_rec2_0 + 8][ii_rec2_1 + 12][ii_rec2_3 + 8] + 3.57142862e-4F*v_z[t0][ii_rec2_0 + 8][ii_rec2_1 + 8][ii_rec2_3 + 4] - 3.80952387e-3F*v_z[t0][ii_rec2_0 + 8][ii_rec2_1 + 8][ii_rec2_3 + 5] + 2.00000003e-2F*v_z[t0][ii_rec2_0 + 8][ii_rec2_1 + 8][ii_rec2_3 + 6] - 8.00000012e-2F*v_z[t0][ii_rec2_0 + 8][ii_rec2_1 + 8][ii_rec2_3 + 7] + 8.00000012e-2F*v_z[t0][ii_rec2_0 + 8][ii_rec2_1 + 8][ii_rec2_3 + 9] - 2.00000003e-2F*v_z[t0][ii_rec2_0 + 8][ii_rec2_1 + 8][ii_rec2_3 + 10] + 3.80952387e-3F*v_z[t0][ii_rec2_0 + 8][ii_rec2_1 + 8][ii_rec2_3 + 11] - 3.57142862e-4F*v_z[t0][ii_rec2_0 + 8][ii_rec2_1 + 8][ii_rec2_3 + 12]);
        }
        if (ii_rec2_0 >= x_m - 1 && ii_rec2_2 >= z_m - 1 && ii_rec2_4 >= y_m - 1 && ii_rec2_0 <= x_M + 1 && ii_rec2_2 <= z_M + 1 && ii_rec2_4 <= y_M + 1)
        {
          sum += (1.0e-3F*px*py*pz - 1.0e-2F*px*py - 1.0e-2F*py*pz + 1.0e-1F*py)*(3.57142862e-4F*v_x[t0][ii_rec2_0 + 4][ii_rec2_4 + 8][ii_rec2_2 + 8] - 3.80952387e-3F*v_x[t0][ii_rec2_0 + 5][ii_rec2_4 + 8][ii_rec2_2 + 8] + 2.00000003e-2F*v_x[t0][ii_rec2_0 + 6][ii_rec2_4 + 8][ii_rec2_2 + 8] - 8.00000012e-2F*v_x[t0][ii_rec2_0 + 7][ii_rec2_4 + 8][ii_rec2_2 + 8] + 8.00000012e-2F*v_x[t0][ii_rec2_0 + 9][ii_rec2_4 + 8][ii_rec2_2 + 8] - 2.00000003e-2F*v_x[t0][ii_rec2_0 + 10][ii_rec2_4 + 8][ii_rec2_2 + 8] + 3.80952387e-3F*v_x[t0][ii_rec2_0 + 11][ii_rec2_4 + 8][ii_rec2_2 + 8] - 3.57142862e-4F*v_x[t0][ii_rec2_0 + 12][ii_rec2_4 + 8][ii_rec2_2 + 8] + 3.57142862e-4F*v_y[t0][ii_rec2_0 + 8][ii_rec2_4 + 4][ii_rec2_2 + 8] - 3.80952387e-3F*v_y[t0][ii_rec2_0 + 8][ii_rec2_4 + 5][ii_rec2_2 + 8] + 2.00000003e-2F*v_y[t0][ii_rec2_0 + 8][ii_rec2_4 + 6][ii_rec2_2 + 8] - 8.00000012e-2F*v_y[t0][ii_rec2_0 + 8][ii_rec2_4 + 7][ii_rec2_2 + 8] + 8.00000012e-2F*v_y[t0][ii_rec2_0 + 8][ii_rec2_4 + 9][ii_rec2_2 + 8] - 2.00000003e-2F*v_y[t0][ii_rec2_0 + 8][ii_rec2_4 + 10][ii_rec2_2 + 8] + 3.80952387e-3F*v_y[t0][ii_rec2_0 + 8][ii_rec2_4 + 11][ii_rec2_2 + 8] - 3.57142862e-4F*v_y[t0][ii_rec2_0 + 8][ii_rec2_4 + 12][ii_rec2_2 + 8] + 3.57142862e-4F*v_z[t0][ii_rec2_0 + 8][ii_rec2_4 + 8][ii_rec2_2 + 4] - 3.80952387e-3F*v_z[t0][ii_rec2_0 + 8][ii_rec2_4 + 8][ii_rec2_2 + 5] + 2.00000003e-2F*v_z[t0][ii_rec2_0 + 8][ii_rec2_4 + 8][ii_rec2_2 + 6] - 8.00000012e-2F*v_z[t0][ii_rec2_0 + 8][ii_rec2_4 + 8][ii_rec2_2 + 7] + 8.00000012e-2F*v_z[t0][ii_rec2_0 + 8][ii_rec2_4 + 8][ii_rec2_2 + 9] - 2.00000003e-2F*v_z[t0][ii_rec2_0 + 8][ii_rec2_4 + 8][ii_rec2_2 + 10] + 3.80952387e-3F*v_z[t0][ii_rec2_0 + 8][ii_rec2_4 + 8][ii_rec2_2 + 11] - 3.57142862e-4F*v_z[t0][ii_rec2_0 + 8][ii_rec2_4 + 8][ii_rec2_2 + 12]);
        }
        if (ii_rec2_0 >= x_m - 1 && ii_rec2_3 >= z_m - 1 && ii_rec2_4 >= y_m - 1 && ii_rec2_0 <= x_M + 1 && ii_rec2_3 <= z_M + 1 && ii_rec2_4 <= y_M + 1)
        {
          sum += (-1.0e-3F*px*py*pz + 1.0e-2F*py*pz)*(3.57142862e-4F*v_x[t0][ii_rec2_0 + 4][ii_rec2_4 + 8][ii_rec2_3 + 8] - 3.80952387e-3F*v_x[t0][ii_rec2_0 + 5][ii_rec2_4 + 8][ii_rec2_3 + 8] + 2.00000003e-2F*v_x[t0][ii_rec2_0 + 6][ii_rec2_4 + 8][ii_rec2_3 + 8] - 8.00000012e-2F*v_x[t0][ii_rec2_0 + 7][ii_rec2_4 + 8][ii_rec2_3 + 8] + 8.00000012e-2F*v_x[t0][ii_rec2_0 + 9][ii_rec2_4 + 8][ii_rec2_3 + 8] - 2.00000003e-2F*v_x[t0][ii_rec2_0 + 10][ii_rec2_4 + 8][ii_rec2_3 + 8] + 3.80952387e-3F*v_x[t0][ii_rec2_0 + 11][ii_rec2_4 + 8][ii_rec2_3 + 8] - 3.57142862e-4F*v_x[t0][ii_rec2_0 + 12][ii_rec2_4 + 8][ii_rec2_3 + 8] + 3.57142862e-4F*v_y[t0][ii_rec2_0 + 8][ii_rec2_4 + 4][ii_rec2_3 + 8] - 3.80952387e-3F*v_y[t0][ii_rec2_0 + 8][ii_rec2_4 + 5][ii_rec2_3 + 8] + 2.00000003e-2F*v_y[t0][ii_rec2_0 + 8][ii_rec2_4 + 6][ii_rec2_3 + 8] - 8.00000012e-2F*v_y[t0][ii_rec2_0 + 8][ii_rec2_4 + 7][ii_rec2_3 + 8] + 8.00000012e-2F*v_y[t0][ii_rec2_0 + 8][ii_rec2_4 + 9][ii_rec2_3 + 8] - 2.00000003e-2F*v_y[t0][ii_rec2_0 + 8][ii_rec2_4 + 10][ii_rec2_3 + 8] + 3.80952387e-3F*v_y[t0][ii_rec2_0 + 8][ii_rec2_4 + 11][ii_rec2_3 + 8] - 3.57142862e-4F*v_y[t0][ii_rec2_0 + 8][ii_rec2_4 + 12][ii_rec2_3 + 8] + 3.57142862e-4F*v_z[t0][ii_rec2_0 + 8][ii_rec2_4 + 8][ii_rec2_3 + 4] - 3.80952387e-3F*v_z[t0][ii_rec2_0 + 8][ii_rec2_4 + 8][ii_rec2_3 + 5] + 2.00000003e-2F*v_z[t0][ii_rec2_0 + 8][ii_rec2_4 + 8][ii_rec2_3 + 6] - 8.00000012e-2F*v_z[t0][ii_rec2_0 + 8][ii_rec2_4 + 8][ii_rec2_3 + 7] + 8.00000012e-2F*v_z[t0][ii_rec2_0 + 8][ii_rec2_4 + 8][ii_rec2_3 + 9] - 2.00000003e-2F*v_z[t0][ii_rec2_0 + 8][ii_rec2_4 + 8][ii_rec2_3 + 10] + 3.80952387e-3F*v_z[t0][ii_rec2_0 + 8][ii_rec2_4 + 8][ii_rec2_3 + 11] - 3.57142862e-4F*v_z[t0][ii_rec2_0 + 8][ii_rec2_4 + 8][ii_rec2_3 + 12]);
        }
        if (ii_rec2_1 >= y_m - 1 && ii_rec2_2 >= z_m - 1 && ii_rec2_5 >= x_m - 1 && ii_rec2_1 <= y_M + 1 && ii_rec2_2 <= z_M + 1 && ii_rec2_5 <= x_M + 1)
        {
          sum += (1.0e-3F*px*py*pz - 1.0e-2F*px*py - 1.0e-2F*px*pz + 1.0e-1F*px)*(3.57142862e-4F*v_x[t0][ii_rec2_5 + 4][ii_rec2_1 + 8][ii_rec2_2 + 8] - 3.80952387e-3F*v_x[t0][ii_rec2_5 + 5][ii_rec2_1 + 8][ii_rec2_2 + 8] + 2.00000003e-2F*v_x[t0][ii_rec2_5 + 6][ii_rec2_1 + 8][ii_rec2_2 + 8] - 8.00000012e-2F*v_x[t0][ii_rec2_5 + 7][ii_rec2_1 + 8][ii_rec2_2 + 8] + 8.00000012e-2F*v_x[t0][ii_rec2_5 + 9][ii_rec2_1 + 8][ii_rec2_2 + 8] - 2.00000003e-2F*v_x[t0][ii_rec2_5 + 10][ii_rec2_1 + 8][ii_rec2_2 + 8] + 3.80952387e-3F*v_x[t0][ii_rec2_5 + 11][ii_rec2_1 + 8][ii_rec2_2 + 8] - 3.57142862e-4F*v_x[t0][ii_rec2_5 + 12][ii_rec2_1 + 8][ii_rec2_2 + 8] + 3.57142862e-4F*v_y[t0][ii_rec2_5 + 8][ii_rec2_1 + 4][ii_rec2_2 + 8] - 3.80952387e-3F*v_y[t0][ii_rec2_5 + 8][ii_rec2_1 + 5][ii_rec2_2 + 8] + 2.00000003e-2F*v_y[t0][ii_rec2_5 + 8][ii_rec2_1 + 6][ii_rec2_2 + 8] - 8.00000012e-2F*v_y[t0][ii_rec2_5 + 8][ii_rec2_1 + 7][ii_rec2_2 + 8] + 8.00000012e-2F*v_y[t0][ii_rec2_5 + 8][ii_rec2_1 + 9][ii_rec2_2 + 8] - 2.00000003e-2F*v_y[t0][ii_rec2_5 + 8][ii_rec2_1 + 10][ii_rec2_2 + 8] + 3.80952387e-3F*v_y[t0][ii_rec2_5 + 8][ii_rec2_1 + 11][ii_rec2_2 + 8] - 3.57142862e-4F*v_y[t0][ii_rec2_5 + 8][ii_rec2_1 + 12][ii_rec2_2 + 8] + 3.57142862e-4F*v_z[t0][ii_rec2_5 + 8][ii_rec2_1 + 8][ii_rec2_2 + 4] - 3.80952387e-3F*v_z[t0][ii_rec2_5 + 8][ii_rec2_1 + 8][ii_rec2_2 + 5] + 2.00000003e-2F*v_z[t0][ii_rec2_5 + 8][ii_rec2_1 + 8][ii_rec2_2 + 6] - 8.00000012e-2F*v_z[t0][ii_rec2_5 + 8][ii_rec2_1 + 8][ii_rec2_2 + 7] + 8.00000012e-2F*v_z[t0][ii_rec2_5 + 8][ii_rec2_1 + 8][ii_rec2_2 + 9] - 2.00000003e-2F*v_z[t0][ii_rec2_5 + 8][ii_rec2_1 + 8][ii_rec2_2 + 10] + 3.80952387e-3F*v_z[t0][ii_rec2_5 + 8][ii_rec2_1 + 8][ii_rec2_2 + 11] - 3.57142862e-4F*v_z[t0][ii_rec2_5 + 8][ii_rec2_1 + 8][ii_rec2_2 + 12]);
        }
        if (ii_rec2_1 >= y_m - 1 && ii_rec2_3 >= z_m - 1 && ii_rec2_5 >= x_m - 1 && ii_rec2_1 <= y_M + 1 && ii_rec2_3 <= z_M + 1 && ii_rec2_5 <= x_M + 1)
        {
          sum += (-1.0e-3F*px*py*pz + 1.0e-2F*px*pz)*(3.57142862e-4F*v_x[t0][ii_rec2_5 + 4][ii_rec2_1 + 8][ii_rec2_3 + 8] - 3.80952387e-3F*v_x[t0][ii_rec2_5 + 5][ii_rec2_1 + 8][ii_rec2_3 + 8] + 2.00000003e-2F*v_x[t0][ii_rec2_5 + 6][ii_rec2_1 + 8][ii_rec2_3 + 8] - 8.00000012e-2F*v_x[t0][ii_rec2_5 + 7][ii_rec2_1 + 8][ii_rec2_3 + 8] + 8.00000012e-2F*v_x[t0][ii_rec2_5 + 9][ii_rec2_1 + 8][ii_rec2_3 + 8] - 2.00000003e-2F*v_x[t0][ii_rec2_5 + 10][ii_rec2_1 + 8][ii_rec2_3 + 8] + 3.80952387e-3F*v_x[t0][ii_rec2_5 + 11][ii_rec2_1 + 8][ii_rec2_3 + 8] - 3.57142862e-4F*v_x[t0][ii_rec2_5 + 12][ii_rec2_1 + 8][ii_rec2_3 + 8] + 3.57142862e-4F*v_y[t0][ii_rec2_5 + 8][ii_rec2_1 + 4][ii_rec2_3 + 8] - 3.80952387e-3F*v_y[t0][ii_rec2_5 + 8][ii_rec2_1 + 5][ii_rec2_3 + 8] + 2.00000003e-2F*v_y[t0][ii_rec2_5 + 8][ii_rec2_1 + 6][ii_rec2_3 + 8] - 8.00000012e-2F*v_y[t0][ii_rec2_5 + 8][ii_rec2_1 + 7][ii_rec2_3 + 8] + 8.00000012e-2F*v_y[t0][ii_rec2_5 + 8][ii_rec2_1 + 9][ii_rec2_3 + 8] - 2.00000003e-2F*v_y[t0][ii_rec2_5 + 8][ii_rec2_1 + 10][ii_rec2_3 + 8] + 3.80952387e-3F*v_y[t0][ii_rec2_5 + 8][ii_rec2_1 + 11][ii_rec2_3 + 8] - 3.57142862e-4F*v_y[t0][ii_rec2_5 + 8][ii_rec2_1 + 12][ii_rec2_3 + 8] + 3.57142862e-4F*v_z[t0][ii_rec2_5 + 8][ii_rec2_1 + 8][ii_rec2_3 + 4] - 3.80952387e-3F*v_z[t0][ii_rec2_5 + 8][ii_rec2_1 + 8][ii_rec2_3 + 5] + 2.00000003e-2F*v_z[t0][ii_rec2_5 + 8][ii_rec2_1 + 8][ii_rec2_3 + 6] - 8.00000012e-2F*v_z[t0][ii_rec2_5 + 8][ii_rec2_1 + 8][ii_rec2_3 + 7] + 8.00000012e-2F*v_z[t0][ii_rec2_5 + 8][ii_rec2_1 + 8][ii_rec2_3 + 9] - 2.00000003e-2F*v_z[t0][ii_rec2_5 + 8][ii_rec2_1 + 8][ii_rec2_3 + 10] + 3.80952387e-3F*v_z[t0][ii_rec2_5 + 8][ii_rec2_1 + 8][ii_rec2_3 + 11] - 3.57142862e-4F*v_z[t0][ii_rec2_5 + 8][ii_rec2_1 + 8][ii_rec2_3 + 12]);
        }
        if (ii_rec2_2 >= z_m - 1 && ii_rec2_4 >= y_m - 1 && ii_rec2_5 >= x_m - 1 && ii_rec2_2 <= z_M + 1 && ii_rec2_4 <= y_M + 1 && ii_rec2_5 <= x_M + 1)
        {
          sum += (-1.0e-3F*px*py*pz + 1.0e-2F*px*py)*(3.57142862e-4F*v_x[t0][ii_rec2_5 + 4][ii_rec2_4 + 8][ii_rec2_2 + 8] - 3.80952387e-3F*v_x[t0][ii_rec2_5 + 5][ii_rec2_4 + 8][ii_rec2_2 + 8] + 2.00000003e-2F*v_x[t0][ii_rec2_5 + 6][ii_rec2_4 + 8][ii_rec2_2 + 8] - 8.00000012e-2F*v_x[t0][ii_rec2_5 + 7][ii_rec2_4 + 8][ii_rec2_2 + 8] + 8.00000012e-2F*v_x[t0][ii_rec2_5 + 9][ii_rec2_4 + 8][ii_rec2_2 + 8] - 2.00000003e-2F*v_x[t0][ii_rec2_5 + 10][ii_rec2_4 + 8][ii_rec2_2 + 8] + 3.80952387e-3F*v_x[t0][ii_rec2_5 + 11][ii_rec2_4 + 8][ii_rec2_2 + 8] - 3.57142862e-4F*v_x[t0][ii_rec2_5 + 12][ii_rec2_4 + 8][ii_rec2_2 + 8] + 3.57142862e-4F*v_y[t0][ii_rec2_5 + 8][ii_rec2_4 + 4][ii_rec2_2 + 8] - 3.80952387e-3F*v_y[t0][ii_rec2_5 + 8][ii_rec2_4 + 5][ii_rec2_2 + 8] + 2.00000003e-2F*v_y[t0][ii_rec2_5 + 8][ii_rec2_4 + 6][ii_rec2_2 + 8] - 8.00000012e-2F*v_y[t0][ii_rec2_5 + 8][ii_rec2_4 + 7][ii_rec2_2 + 8] + 8.00000012e-2F*v_y[t0][ii_rec2_5 + 8][ii_rec2_4 + 9][ii_rec2_2 + 8] - 2.00000003e-2F*v_y[t0][ii_rec2_5 + 8][ii_rec2_4 + 10][ii_rec2_2 + 8] + 3.80952387e-3F*v_y[t0][ii_rec2_5 + 8][ii_rec2_4 + 11][ii_rec2_2 + 8] - 3.57142862e-4F*v_y[t0][ii_rec2_5 + 8][ii_rec2_4 + 12][ii_rec2_2 + 8] + 3.57142862e-4F*v_z[t0][ii_rec2_5 + 8][ii_rec2_4 + 8][ii_rec2_2 + 4] - 3.80952387e-3F*v_z[t0][ii_rec2_5 + 8][ii_rec2_4 + 8][ii_rec2_2 + 5] + 2.00000003e-2F*v_z[t0][ii_rec2_5 + 8][ii_rec2_4 + 8][ii_rec2_2 + 6] - 8.00000012e-2F*v_z[t0][ii_rec2_5 + 8][ii_rec2_4 + 8][ii_rec2_2 + 7] + 8.00000012e-2F*v_z[t0][ii_rec2_5 + 8][ii_rec2_4 + 8][ii_rec2_2 + 9] - 2.00000003e-2F*v_z[t0][ii_rec2_5 + 8][ii_rec2_4 + 8][ii_rec2_2 + 10] + 3.80952387e-3F*v_z[t0][ii_rec2_5 + 8][ii_rec2_4 + 8][ii_rec2_2 + 11] - 3.57142862e-4F*v_z[t0][ii_rec2_5 + 8][ii_rec2_4 + 8][ii_rec2_2 + 12]);
        }
        if (ii_rec2_3 >= z_m - 1 && ii_rec2_4 >= y_m - 1 && ii_rec2_5 >= x_m - 1 && ii_rec2_3 <= z_M + 1 && ii_rec2_4 <= y_M + 1 && ii_rec2_5 <= x_M + 1)
        {
          sum += 1.0e-3F*px*py*pz*(3.57142862e-4F*v_x[t0][ii_rec2_5 + 4][ii_rec2_4 + 8][ii_rec2_3 + 8] - 3.80952387e-3F*v_x[t0][ii_rec2_5 + 5][ii_rec2_4 + 8][ii_rec2_3 + 8] + 2.00000003e-2F*v_x[t0][ii_rec2_5 + 6][ii_rec2_4 + 8][ii_rec2_3 + 8] - 8.00000012e-2F*v_x[t0][ii_rec2_5 + 7][ii_rec2_4 + 8][ii_rec2_3 + 8] + 8.00000012e-2F*v_x[t0][ii_rec2_5 + 9][ii_rec2_4 + 8][ii_rec2_3 + 8] - 2.00000003e-2F*v_x[t0][ii_rec2_5 + 10][ii_rec2_4 + 8][ii_rec2_3 + 8] + 3.80952387e-3F*v_x[t0][ii_rec2_5 + 11][ii_rec2_4 + 8][ii_rec2_3 + 8] - 3.57142862e-4F*v_x[t0][ii_rec2_5 + 12][ii_rec2_4 + 8][ii_rec2_3 + 8] + 3.57142862e-4F*v_y[t0][ii_rec2_5 + 8][ii_rec2_4 + 4][ii_rec2_3 + 8] - 3.80952387e-3F*v_y[t0][ii_rec2_5 + 8][ii_rec2_4 + 5][ii_rec2_3 + 8] + 2.00000003e-2F*v_y[t0][ii_rec2_5 + 8][ii_rec2_4 + 6][ii_rec2_3 + 8] - 8.00000012e-2F*v_y[t0][ii_rec2_5 + 8][ii_rec2_4 + 7][ii_rec2_3 + 8] + 8.00000012e-2F*v_y[t0][ii_rec2_5 + 8][ii_rec2_4 + 9][ii_rec2_3 + 8] - 2.00000003e-2F*v_y[t0][ii_rec2_5 + 8][ii_rec2_4 + 10][ii_rec2_3 + 8] + 3.80952387e-3F*v_y[t0][ii_rec2_5 + 8][ii_rec2_4 + 11][ii_rec2_3 + 8] - 3.57142862e-4F*v_y[t0][ii_rec2_5 + 8][ii_rec2_4 + 12][ii_rec2_3 + 8] + 3.57142862e-4F*v_z[t0][ii_rec2_5 + 8][ii_rec2_4 + 8][ii_rec2_3 + 4] - 3.80952387e-3F*v_z[t0][ii_rec2_5 + 8][ii_rec2_4 + 8][ii_rec2_3 + 5] + 2.00000003e-2F*v_z[t0][ii_rec2_5 + 8][ii_rec2_4 + 8][ii_rec2_3 + 6] - 8.00000012e-2F*v_z[t0][ii_rec2_5 + 8][ii_rec2_4 + 8][ii_rec2_3 + 7] + 8.00000012e-2F*v_z[t0][ii_rec2_5 + 8][ii_rec2_4 + 8][ii_rec2_3 + 9] - 2.00000003e-2F*v_z[t0][ii_rec2_5 + 8][ii_rec2_4 + 8][ii_rec2_3 + 10] + 3.80952387e-3F*v_z[t0][ii_rec2_5 + 8][ii_rec2_4 + 8][ii_rec2_3 + 11] - 3.57142862e-4F*v_z[t0][ii_rec2_5 + 8][ii_rec2_4 + 8][ii_rec2_3 + 12]);
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

void sendrecv_txyz(struct dataobj *restrict a_vec, const int buf_x_size, const int buf_y_size, const int buf_z_size, int ogtime, int ogx, int ogy, int ogz, int ostime, int osx, int osy, int osz, int fromrank, int torank, MPI_Comm comm, const int nthreads)
{
  float (*restrict a)[a_vec->size[1]][a_vec->size[2]][a_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[a_vec->size[1]][a_vec->size[2]][a_vec->size[3]]) a_vec->data;
  float (*bufs)[buf_y_size][buf_z_size];
  posix_memalign((void**)&bufs, 64, sizeof(float[buf_x_size][buf_y_size][buf_z_size]));
  float (*bufg)[buf_y_size][buf_z_size];
  posix_memalign((void**)&bufg, 64, sizeof(float[buf_x_size][buf_y_size][buf_z_size]));
  MPI_Request rrecv;
  MPI_Request rsend;
  MPI_Irecv((float *)bufs,buf_x_size*buf_y_size*buf_z_size,MPI_FLOAT,fromrank,13,comm,&rrecv);
  if (torank != MPI_PROC_NULL)
  {
    gather_txyz((float *)bufg,buf_x_size,buf_y_size,buf_z_size,a_vec,ogtime,ogx,ogy,ogz,nthreads);
  }
  MPI_Isend((float *)bufg,buf_x_size*buf_y_size*buf_z_size,MPI_FLOAT,torank,13,comm,&rsend);
  MPI_Wait(&rsend,MPI_STATUS_IGNORE);
  MPI_Wait(&rrecv,MPI_STATUS_IGNORE);
  if (fromrank != MPI_PROC_NULL)
  {
    scatter_txyz((float *)bufs,buf_x_size,buf_y_size,buf_z_size,a_vec,ostime,osx,osy,osz,nthreads);
  }
  free(bufs);
  free(bufg);
}

void gather_txyz(float *restrict buf_vec, const int buf_x_size, const int buf_y_size, const int buf_z_size, struct dataobj *restrict a_vec, int otime, int ox, int oy, int oz, const int nthreads)
{
  float (*restrict buf)[buf_y_size][buf_z_size] __attribute__ ((aligned (64))) = (float (*)[buf_y_size][buf_z_size]) buf_vec;
  float (*restrict a)[a_vec->size[1]][a_vec->size[2]][a_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[a_vec->size[1]][a_vec->size[2]][a_vec->size[3]]) a_vec->data;
  #pragma omp parallel num_threads(nthreads)
  {
    #pragma omp for collapse(1) schedule(dynamic,1)
    for (int x = 0; x <= buf_x_size - 1; x += 1)
    {
      for (int y = 0; y <= buf_y_size - 1; y += 1)
      {
        #pragma omp simd aligned(a:32)
        for (int z = 0; z <= buf_z_size - 1; z += 1)
        {
          buf[x][y][z] = a[otime][x + ox][y + oy][z + oz];
        }
      }
    }
  }
}

void scatter_txyz(float *restrict buf_vec, const int buf_x_size, const int buf_y_size, const int buf_z_size, struct dataobj *restrict a_vec, int otime, int ox, int oy, int oz, const int nthreads)
{
  float (*restrict buf)[buf_y_size][buf_z_size] __attribute__ ((aligned (64))) = (float (*)[buf_y_size][buf_z_size]) buf_vec;
  float (*restrict a)[a_vec->size[1]][a_vec->size[2]][a_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[a_vec->size[1]][a_vec->size[2]][a_vec->size[3]]) a_vec->data;
  #pragma omp parallel num_threads(nthreads)
  {
    #pragma omp for collapse(1) schedule(dynamic,1)
    for (int x = 0; x <= buf_x_size - 1; x += 1)
    {
      for (int y = 0; y <= buf_y_size - 1; y += 1)
      {
        #pragma omp simd aligned(a:32)
        for (int z = 0; z <= buf_z_size - 1; z += 1)
        {
          a[otime][x + ox][y + oy][z + oz] = buf[x][y][z];
        }
      }
    }
  }
}

void haloupdate7(struct dataobj *restrict a_vec, MPI_Comm comm, struct neighborhood * nb, int otime, const int nthreads)
{
  float (*restrict a)[a_vec->size[1]][a_vec->size[2]][a_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[a_vec->size[1]][a_vec->size[2]][a_vec->size[3]]) a_vec->data;
  sendrecv_txyz(a_vec,a_vec->hsize[3],a_vec->npsize[2],a_vec->npsize[3],otime,a_vec->oofs[2],a_vec->hofs[4],a_vec->hofs[6],otime,a_vec->hofs[3],a_vec->hofs[4],a_vec->hofs[6],nb->rcc,nb->lcc,comm,nthreads);
  sendrecv_txyz(a_vec,a_vec->hsize[2],a_vec->npsize[2],a_vec->npsize[3],otime,a_vec->oofs[3],a_vec->hofs[4],a_vec->hofs[6],otime,a_vec->hofs[2],a_vec->hofs[4],a_vec->hofs[6],nb->lcc,nb->rcc,comm,nthreads);
  sendrecv_txyz(a_vec,a_vec->npsize[1],a_vec->hsize[5],a_vec->npsize[3],otime,a_vec->hofs[2],a_vec->oofs[4],a_vec->hofs[6],otime,a_vec->hofs[2],a_vec->hofs[5],a_vec->hofs[6],nb->crc,nb->clc,comm,nthreads);
  sendrecv_txyz(a_vec,a_vec->npsize[1],a_vec->hsize[4],a_vec->npsize[3],otime,a_vec->hofs[2],a_vec->oofs[5],a_vec->hofs[6],otime,a_vec->hofs[2],a_vec->hofs[4],a_vec->hofs[6],nb->clc,nb->crc,comm,nthreads);
  sendrecv_txyz(a_vec,a_vec->npsize[1],a_vec->npsize[2],a_vec->hsize[7],otime,a_vec->hofs[2],a_vec->hofs[4],a_vec->oofs[6],otime,a_vec->hofs[2],a_vec->hofs[4],a_vec->hofs[7],nb->ccr,nb->ccl,comm,nthreads);
  sendrecv_txyz(a_vec,a_vec->npsize[1],a_vec->npsize[2],a_vec->hsize[6],otime,a_vec->hofs[2],a_vec->hofs[4],a_vec->oofs[7],otime,a_vec->hofs[2],a_vec->hofs[4],a_vec->hofs[6],nb->ccl,nb->ccr,comm,nthreads);
}

void haloupdate0(struct dataobj *restrict a_vec, MPI_Comm comm, struct neighborhood * nb, int otime, const int nthreads)
{
  float (*restrict a)[a_vec->size[1]][a_vec->size[2]][a_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[a_vec->size[1]][a_vec->size[2]][a_vec->size[3]]) a_vec->data;
  sendrecv_txyz(a_vec,a_vec->hsize[3],a_vec->npsize[2],a_vec->npsize[3],otime,a_vec->oofs[2],a_vec->hofs[4],a_vec->hofs[6],otime,a_vec->hofs[3],a_vec->hofs[4],a_vec->hofs[6],nb->rcc,nb->lcc,comm,nthreads);
  sendrecv_txyz(a_vec,a_vec->hsize[2],a_vec->npsize[2],a_vec->npsize[3],otime,a_vec->oofs[3],a_vec->hofs[4],a_vec->hofs[6],otime,a_vec->hofs[2],a_vec->hofs[4],a_vec->hofs[6],nb->lcc,nb->rcc,comm,nthreads);
}

void haloupdate1(struct dataobj *restrict a_vec, MPI_Comm comm, struct neighborhood * nb, int otime, const int nthreads)
{
  float (*restrict a)[a_vec->size[1]][a_vec->size[2]][a_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[a_vec->size[1]][a_vec->size[2]][a_vec->size[3]]) a_vec->data;
  sendrecv_txyz(a_vec,a_vec->hsize[3],a_vec->npsize[2],a_vec->npsize[3],otime,a_vec->oofs[2],a_vec->hofs[4],a_vec->hofs[6],otime,a_vec->hofs[3],a_vec->hofs[4],a_vec->hofs[6],nb->rcc,nb->lcc,comm,nthreads);
  sendrecv_txyz(a_vec,a_vec->hsize[2],a_vec->npsize[2],a_vec->npsize[3],otime,a_vec->oofs[3],a_vec->hofs[4],a_vec->hofs[6],otime,a_vec->hofs[2],a_vec->hofs[4],a_vec->hofs[6],nb->lcc,nb->rcc,comm,nthreads);
  sendrecv_txyz(a_vec,a_vec->npsize[1],a_vec->hsize[5],a_vec->npsize[3],otime,a_vec->hofs[2],a_vec->oofs[4],a_vec->hofs[6],otime,a_vec->hofs[2],a_vec->hofs[5],a_vec->hofs[6],nb->crc,nb->clc,comm,nthreads);
  sendrecv_txyz(a_vec,a_vec->npsize[1],a_vec->hsize[4],a_vec->npsize[3],otime,a_vec->hofs[2],a_vec->oofs[5],a_vec->hofs[6],otime,a_vec->hofs[2],a_vec->hofs[4],a_vec->hofs[6],nb->clc,nb->crc,comm,nthreads);
}

void haloupdate2(struct dataobj *restrict a_vec, MPI_Comm comm, struct neighborhood * nb, int otime, const int nthreads)
{
  float (*restrict a)[a_vec->size[1]][a_vec->size[2]][a_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[a_vec->size[1]][a_vec->size[2]][a_vec->size[3]]) a_vec->data;
  sendrecv_txyz(a_vec,a_vec->hsize[3],a_vec->npsize[2],a_vec->npsize[3],otime,a_vec->oofs[2],a_vec->hofs[4],a_vec->hofs[6],otime,a_vec->hofs[3],a_vec->hofs[4],a_vec->hofs[6],nb->rcc,nb->lcc,comm,nthreads);
  sendrecv_txyz(a_vec,a_vec->hsize[2],a_vec->npsize[2],a_vec->npsize[3],otime,a_vec->oofs[3],a_vec->hofs[4],a_vec->hofs[6],otime,a_vec->hofs[2],a_vec->hofs[4],a_vec->hofs[6],nb->lcc,nb->rcc,comm,nthreads);
  sendrecv_txyz(a_vec,a_vec->npsize[1],a_vec->npsize[2],a_vec->hsize[7],otime,a_vec->hofs[2],a_vec->hofs[4],a_vec->oofs[6],otime,a_vec->hofs[2],a_vec->hofs[4],a_vec->hofs[7],nb->ccr,nb->ccl,comm,nthreads);
  sendrecv_txyz(a_vec,a_vec->npsize[1],a_vec->npsize[2],a_vec->hsize[6],otime,a_vec->hofs[2],a_vec->hofs[4],a_vec->oofs[7],otime,a_vec->hofs[2],a_vec->hofs[4],a_vec->hofs[6],nb->ccl,nb->ccr,comm,nthreads);
}

void haloupdate3(struct dataobj *restrict a_vec, MPI_Comm comm, struct neighborhood * nb, int otime, const int nthreads)
{
  float (*restrict a)[a_vec->size[1]][a_vec->size[2]][a_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[a_vec->size[1]][a_vec->size[2]][a_vec->size[3]]) a_vec->data;
  sendrecv_txyz(a_vec,a_vec->npsize[1],a_vec->hsize[5],a_vec->npsize[3],otime,a_vec->hofs[2],a_vec->oofs[4],a_vec->hofs[6],otime,a_vec->hofs[2],a_vec->hofs[5],a_vec->hofs[6],nb->crc,nb->clc,comm,nthreads);
  sendrecv_txyz(a_vec,a_vec->npsize[1],a_vec->hsize[4],a_vec->npsize[3],otime,a_vec->hofs[2],a_vec->oofs[5],a_vec->hofs[6],otime,a_vec->hofs[2],a_vec->hofs[4],a_vec->hofs[6],nb->clc,nb->crc,comm,nthreads);
}

void haloupdate4(struct dataobj *restrict a_vec, MPI_Comm comm, struct neighborhood * nb, int otime, const int nthreads)
{
  float (*restrict a)[a_vec->size[1]][a_vec->size[2]][a_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[a_vec->size[1]][a_vec->size[2]][a_vec->size[3]]) a_vec->data;
  sendrecv_txyz(a_vec,a_vec->npsize[1],a_vec->hsize[5],a_vec->npsize[3],otime,a_vec->hofs[2],a_vec->oofs[4],a_vec->hofs[6],otime,a_vec->hofs[2],a_vec->hofs[5],a_vec->hofs[6],nb->crc,nb->clc,comm,nthreads);
  sendrecv_txyz(a_vec,a_vec->npsize[1],a_vec->hsize[4],a_vec->npsize[3],otime,a_vec->hofs[2],a_vec->oofs[5],a_vec->hofs[6],otime,a_vec->hofs[2],a_vec->hofs[4],a_vec->hofs[6],nb->clc,nb->crc,comm,nthreads);
  sendrecv_txyz(a_vec,a_vec->npsize[1],a_vec->npsize[2],a_vec->hsize[7],otime,a_vec->hofs[2],a_vec->hofs[4],a_vec->oofs[6],otime,a_vec->hofs[2],a_vec->hofs[4],a_vec->hofs[7],nb->ccr,nb->ccl,comm,nthreads);
  sendrecv_txyz(a_vec,a_vec->npsize[1],a_vec->npsize[2],a_vec->hsize[6],otime,a_vec->hofs[2],a_vec->hofs[4],a_vec->oofs[7],otime,a_vec->hofs[2],a_vec->hofs[4],a_vec->hofs[6],nb->ccl,nb->ccr,comm,nthreads);
}

void haloupdate5(struct dataobj *restrict a_vec, MPI_Comm comm, struct neighborhood * nb, int otime, const int nthreads)
{
  float (*restrict a)[a_vec->size[1]][a_vec->size[2]][a_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[a_vec->size[1]][a_vec->size[2]][a_vec->size[3]]) a_vec->data;
  sendrecv_txyz(a_vec,a_vec->hsize[3],a_vec->npsize[2],a_vec->npsize[3],otime,a_vec->oofs[2],a_vec->hofs[4],a_vec->hofs[6],otime,a_vec->hofs[3],a_vec->hofs[4],a_vec->hofs[6],nb->rcc,nb->lcc,comm,nthreads);
  sendrecv_txyz(a_vec,a_vec->hsize[2],a_vec->npsize[2],a_vec->npsize[3],otime,a_vec->oofs[3],a_vec->hofs[4],a_vec->hofs[6],otime,a_vec->hofs[2],a_vec->hofs[4],a_vec->hofs[6],nb->lcc,nb->rcc,comm,nthreads);
  sendrecv_txyz(a_vec,a_vec->npsize[1],a_vec->hsize[5],a_vec->npsize[3],otime,a_vec->hofs[2],a_vec->oofs[4],a_vec->hofs[6],otime,a_vec->hofs[2],a_vec->hofs[5],a_vec->hofs[6],nb->crc,nb->clc,comm,nthreads);
  sendrecv_txyz(a_vec,a_vec->npsize[1],a_vec->hsize[4],a_vec->npsize[3],otime,a_vec->hofs[2],a_vec->oofs[5],a_vec->hofs[6],otime,a_vec->hofs[2],a_vec->hofs[4],a_vec->hofs[6],nb->clc,nb->crc,comm,nthreads);
  sendrecv_txyz(a_vec,a_vec->npsize[1],a_vec->npsize[2],a_vec->hsize[7],otime,a_vec->hofs[2],a_vec->hofs[4],a_vec->oofs[6],otime,a_vec->hofs[2],a_vec->hofs[4],a_vec->hofs[7],nb->ccr,nb->ccl,comm,nthreads);
  sendrecv_txyz(a_vec,a_vec->npsize[1],a_vec->npsize[2],a_vec->hsize[6],otime,a_vec->hofs[2],a_vec->hofs[4],a_vec->oofs[7],otime,a_vec->hofs[2],a_vec->hofs[4],a_vec->hofs[6],nb->ccl,nb->ccr,comm,nthreads);
}

void haloupdate6(struct dataobj *restrict a_vec, MPI_Comm comm, struct neighborhood * nb, int otime, const int nthreads)
{
  float (*restrict a)[a_vec->size[1]][a_vec->size[2]][a_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[a_vec->size[1]][a_vec->size[2]][a_vec->size[3]]) a_vec->data;
  sendrecv_txyz(a_vec,a_vec->hsize[3],a_vec->npsize[2],a_vec->npsize[3],otime,a_vec->oofs[2],a_vec->hofs[4],a_vec->hofs[6],otime,a_vec->hofs[3],a_vec->hofs[4],a_vec->hofs[6],nb->rcc,nb->lcc,comm,nthreads);
  sendrecv_txyz(a_vec,a_vec->hsize[2],a_vec->npsize[2],a_vec->npsize[3],otime,a_vec->oofs[3],a_vec->hofs[4],a_vec->hofs[6],otime,a_vec->hofs[2],a_vec->hofs[4],a_vec->hofs[6],nb->lcc,nb->rcc,comm,nthreads);
  sendrecv_txyz(a_vec,a_vec->npsize[1],a_vec->hsize[5],a_vec->npsize[3],otime,a_vec->hofs[2],a_vec->oofs[4],a_vec->hofs[6],otime,a_vec->hofs[2],a_vec->hofs[5],a_vec->hofs[6],nb->crc,nb->clc,comm,nthreads);
  sendrecv_txyz(a_vec,a_vec->npsize[1],a_vec->hsize[4],a_vec->npsize[3],otime,a_vec->hofs[2],a_vec->oofs[5],a_vec->hofs[6],otime,a_vec->hofs[2],a_vec->hofs[4],a_vec->hofs[6],nb->clc,nb->crc,comm,nthreads);
  sendrecv_txyz(a_vec,a_vec->npsize[1],a_vec->npsize[2],a_vec->hsize[7],otime,a_vec->hofs[2],a_vec->hofs[4],a_vec->oofs[6],otime,a_vec->hofs[2],a_vec->hofs[4],a_vec->hofs[7],nb->ccr,nb->ccl,comm,nthreads);
  sendrecv_txyz(a_vec,a_vec->npsize[1],a_vec->npsize[2],a_vec->hsize[6],otime,a_vec->hofs[2],a_vec->hofs[4],a_vec->oofs[7],otime,a_vec->hofs[2],a_vec->hofs[4],a_vec->hofs[6],nb->ccl,nb->ccr,comm,nthreads);
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
              v_x[t1][x + 8][y + 8][z + 8] = 7.00999975204468e-1F*(irho[x + 8][y + 8][z + 8] + irho[x + 9][y + 8][z + 8])*(6.97544653e-5F*(tau_xx[t0][x + 5][y + 8][z + 8] - tau_xx[t0][x + 12][y + 8][z + 8] + tau_xy[t0][x + 8][y + 4][z + 8] - tau_xy[t0][x + 8][y + 11][z + 8] + tau_xz[t0][x + 8][y + 8][z + 4] - tau_xz[t0][x + 8][y + 8][z + 11]) + 9.57031264e-4F*(-tau_xx[t0][x + 6][y + 8][z + 8] + tau_xx[t0][x + 11][y + 8][z + 8] - tau_xy[t0][x + 8][y + 5][z + 8] + tau_xy[t0][x + 8][y + 10][z + 8] - tau_xz[t0][x + 8][y + 8][z + 5] + tau_xz[t0][x + 8][y + 8][z + 10]) + 7.97526054e-3F*(tau_xx[t0][x + 7][y + 8][z + 8] - tau_xx[t0][x + 10][y + 8][z + 8] + tau_xy[t0][x + 8][y + 6][z + 8] - tau_xy[t0][x + 8][y + 9][z + 8] + tau_xz[t0][x + 8][y + 8][z + 6] - tau_xz[t0][x + 8][y + 8][z + 9]) + 1.19628908e-1F*(-tau_xx[t0][x + 8][y + 8][z + 8] + tau_xx[t0][x + 9][y + 8][z + 8] - tau_xy[t0][x + 8][y + 7][z + 8] + tau_xy[t0][x + 8][y + 8][z + 8] - tau_xz[t0][x + 8][y + 8][z + 7] + tau_xz[t0][x + 8][y + 8][z + 8]))*damp[x + 1][y + 1][z + 1] + damp[x + 1][y + 1][z + 1]*v_x[t0][x + 8][y + 8][z + 8];
              v_y[t1][x + 8][y + 8][z + 8] = 7.00999975204468e-1F*(irho[x + 8][y + 8][z + 8] + irho[x + 8][y + 9][z + 8])*(6.97544653e-5F*(tau_xy[t0][x + 4][y + 8][z + 8] - tau_xy[t0][x + 11][y + 8][z + 8] + tau_yy[t0][x + 8][y + 5][z + 8] - tau_yy[t0][x + 8][y + 12][z + 8] + tau_yz[t0][x + 8][y + 8][z + 4] - tau_yz[t0][x + 8][y + 8][z + 11]) + 9.57031264e-4F*(-tau_xy[t0][x + 5][y + 8][z + 8] + tau_xy[t0][x + 10][y + 8][z + 8] - tau_yy[t0][x + 8][y + 6][z + 8] + tau_yy[t0][x + 8][y + 11][z + 8] - tau_yz[t0][x + 8][y + 8][z + 5] + tau_yz[t0][x + 8][y + 8][z + 10]) + 7.97526054e-3F*(tau_xy[t0][x + 6][y + 8][z + 8] - tau_xy[t0][x + 9][y + 8][z + 8] + tau_yy[t0][x + 8][y + 7][z + 8] - tau_yy[t0][x + 8][y + 10][z + 8] + tau_yz[t0][x + 8][y + 8][z + 6] - tau_yz[t0][x + 8][y + 8][z + 9]) + 1.19628908e-1F*(-tau_xy[t0][x + 7][y + 8][z + 8] + tau_xy[t0][x + 8][y + 8][z + 8] - tau_yy[t0][x + 8][y + 8][z + 8] + tau_yy[t0][x + 8][y + 9][z + 8] - tau_yz[t0][x + 8][y + 8][z + 7] + tau_yz[t0][x + 8][y + 8][z + 8]))*damp[x + 1][y + 1][z + 1] + damp[x + 1][y + 1][z + 1]*v_y[t0][x + 8][y + 8][z + 8];
              v_z[t1][x + 8][y + 8][z + 8] = 7.00999975204468e-1F*(irho[x + 8][y + 8][z + 8] + irho[x + 8][y + 8][z + 9])*(6.97544653e-5F*(tau_xz[t0][x + 4][y + 8][z + 8] - tau_xz[t0][x + 11][y + 8][z + 8] + tau_yz[t0][x + 8][y + 4][z + 8] - tau_yz[t0][x + 8][y + 11][z + 8] + tau_zz[t0][x + 8][y + 8][z + 5] - tau_zz[t0][x + 8][y + 8][z + 12]) + 9.57031264e-4F*(-tau_xz[t0][x + 5][y + 8][z + 8] + tau_xz[t0][x + 10][y + 8][z + 8] - tau_yz[t0][x + 8][y + 5][z + 8] + tau_yz[t0][x + 8][y + 10][z + 8] - tau_zz[t0][x + 8][y + 8][z + 6] + tau_zz[t0][x + 8][y + 8][z + 11]) + 7.97526054e-3F*(tau_xz[t0][x + 6][y + 8][z + 8] - tau_xz[t0][x + 9][y + 8][z + 8] + tau_yz[t0][x + 8][y + 6][z + 8] - tau_yz[t0][x + 8][y + 9][z + 8] + tau_zz[t0][x + 8][y + 8][z + 7] - tau_zz[t0][x + 8][y + 8][z + 10]) + 1.19628908e-1F*(-tau_xz[t0][x + 7][y + 8][z + 8] + tau_xz[t0][x + 8][y + 8][z + 8] - tau_yz[t0][x + 8][y + 7][z + 8] + tau_yz[t0][x + 8][y + 8][z + 8] - tau_zz[t0][x + 8][y + 8][z + 8] + tau_zz[t0][x + 8][y + 8][z + 9]))*damp[x + 1][y + 1][z + 1] + damp[x + 1][y + 1][z + 1]*v_z[t0][x + 8][y + 8][z + 8];
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
              float r55 = -v_z[t1][x + 8][y + 8][z + 8];
              float r54 = -v_y[t1][x + 8][y + 8][z + 8];
              float r53 = -v_x[t1][x + 8][y + 8][z + 8];
              float r52 = -v_z[t1][x + 8][y + 8][z + 9];
              float r51 = -v_y[t1][x + 8][y + 9][z + 8];
              float r50 = -v_x[t1][x + 9][y + 8][z + 8];
              float r49 = -v_z[t1][x + 8][y + 8][z + 5];
              float r48 = -v_y[t1][x + 8][y + 5][z + 8];
              float r47 = -v_x[t1][x + 5][y + 8][z + 8];
              float r46 = -v_z[t1][x + 8][y + 8][z + 7];
              float r45 = -v_y[t1][x + 8][y + 7][z + 8];
              float r44 = -v_x[t1][x + 7][y + 8][z + 8];
              float r43 = -v_z[t1][x + 8][y + 8][z + 11];
              float r42 = -v_y[t1][x + 8][y + 11][z + 8];
              float r41 = -v_x[t1][x + 11][y + 8][z + 8];
              float r40 = 1.402F*(6.97544653e-5F*(r41 + r42 + r43 + v_x[t1][x + 4][y + 8][z + 8] + v_y[t1][x + 8][y + 4][z + 8] + v_z[t1][x + 8][y + 8][z + 4]) + 1.19628908e-1F*(r44 + r45 + r46 + v_x[t1][x + 8][y + 8][z + 8] + v_y[t1][x + 8][y + 8][z + 8] + v_z[t1][x + 8][y + 8][z + 8]) + 9.57031264e-4F*(r47 + r48 + r49 + v_x[t1][x + 10][y + 8][z + 8] + v_y[t1][x + 8][y + 10][z + 8] + v_z[t1][x + 8][y + 8][z + 10]) + 7.97526054e-3F*(r50 + r51 + r52 + v_x[t1][x + 6][y + 8][z + 8] + v_y[t1][x + 8][y + 6][z + 8] + v_z[t1][x + 8][y + 8][z + 6]))*damp[x + 1][y + 1][z + 1]*lam[x + 8][y + 8][z + 8];
              tau_xx[t1][x + 8][y + 8][z + 8] = r40 + 2.804F*(6.97544653e-5F*(r41 + v_x[t1][x + 4][y + 8][z + 8]) + 1.19628908e-1F*(r44 + v_x[t1][x + 8][y + 8][z + 8]) + 9.57031264e-4F*(r47 + v_x[t1][x + 10][y + 8][z + 8]) + 7.97526054e-3F*(r50 + v_x[t1][x + 6][y + 8][z + 8]))*damp[x + 1][y + 1][z + 1]*mu[x + 8][y + 8][z + 8] + damp[x + 1][y + 1][z + 1]*tau_xx[t0][x + 8][y + 8][z + 8];
              tau_xy[t1][x + 8][y + 8][z + 8] = 3.50499987602234e-1F*(1.19628908e-1F*(r53 + r54 + v_x[t1][x + 8][y + 9][z + 8] + v_y[t1][x + 9][y + 8][z + 8]) + 6.97544653e-5F*(v_x[t1][x + 8][y + 5][z + 8] - v_x[t1][x + 8][y + 12][z + 8] + v_y[t1][x + 5][y + 8][z + 8] - v_y[t1][x + 12][y + 8][z + 8]) + 9.57031264e-4F*(-v_x[t1][x + 8][y + 6][z + 8] + v_x[t1][x + 8][y + 11][z + 8] - v_y[t1][x + 6][y + 8][z + 8] + v_y[t1][x + 11][y + 8][z + 8]) + 7.97526054e-3F*(v_x[t1][x + 8][y + 7][z + 8] - v_x[t1][x + 8][y + 10][z + 8] + v_y[t1][x + 7][y + 8][z + 8] - v_y[t1][x + 10][y + 8][z + 8]))*(mu[x + 8][y + 8][z + 8] + mu[x + 8][y + 9][z + 8] + mu[x + 9][y + 8][z + 8] + mu[x + 9][y + 9][z + 8])*damp[x + 1][y + 1][z + 1] + damp[x + 1][y + 1][z + 1]*tau_xy[t0][x + 8][y + 8][z + 8];
              tau_xz[t1][x + 8][y + 8][z + 8] = 3.50499987602234e-1F*(1.19628908e-1F*(r53 + r55 + v_x[t1][x + 8][y + 8][z + 9] + v_z[t1][x + 9][y + 8][z + 8]) + 6.97544653e-5F*(v_x[t1][x + 8][y + 8][z + 5] - v_x[t1][x + 8][y + 8][z + 12] + v_z[t1][x + 5][y + 8][z + 8] - v_z[t1][x + 12][y + 8][z + 8]) + 9.57031264e-4F*(-v_x[t1][x + 8][y + 8][z + 6] + v_x[t1][x + 8][y + 8][z + 11] - v_z[t1][x + 6][y + 8][z + 8] + v_z[t1][x + 11][y + 8][z + 8]) + 7.97526054e-3F*(v_x[t1][x + 8][y + 8][z + 7] - v_x[t1][x + 8][y + 8][z + 10] + v_z[t1][x + 7][y + 8][z + 8] - v_z[t1][x + 10][y + 8][z + 8]))*(mu[x + 8][y + 8][z + 8] + mu[x + 8][y + 8][z + 9] + mu[x + 9][y + 8][z + 8] + mu[x + 9][y + 8][z + 9])*damp[x + 1][y + 1][z + 1] + damp[x + 1][y + 1][z + 1]*tau_xz[t0][x + 8][y + 8][z + 8];
              tau_yy[t1][x + 8][y + 8][z + 8] = r40 + 2.804F*(6.97544653e-5F*(r42 + v_y[t1][x + 8][y + 4][z + 8]) + 1.19628908e-1F*(r45 + v_y[t1][x + 8][y + 8][z + 8]) + 9.57031264e-4F*(r48 + v_y[t1][x + 8][y + 10][z + 8]) + 7.97526054e-3F*(r51 + v_y[t1][x + 8][y + 6][z + 8]))*damp[x + 1][y + 1][z + 1]*mu[x + 8][y + 8][z + 8] + damp[x + 1][y + 1][z + 1]*tau_yy[t0][x + 8][y + 8][z + 8];
              tau_yz[t1][x + 8][y + 8][z + 8] = 3.50499987602234e-1F*(1.19628908e-1F*(r54 + r55 + v_y[t1][x + 8][y + 8][z + 9] + v_z[t1][x + 8][y + 9][z + 8]) + 6.97544653e-5F*(v_y[t1][x + 8][y + 8][z + 5] - v_y[t1][x + 8][y + 8][z + 12] + v_z[t1][x + 8][y + 5][z + 8] - v_z[t1][x + 8][y + 12][z + 8]) + 9.57031264e-4F*(-v_y[t1][x + 8][y + 8][z + 6] + v_y[t1][x + 8][y + 8][z + 11] - v_z[t1][x + 8][y + 6][z + 8] + v_z[t1][x + 8][y + 11][z + 8]) + 7.97526054e-3F*(v_y[t1][x + 8][y + 8][z + 7] - v_y[t1][x + 8][y + 8][z + 10] + v_z[t1][x + 8][y + 7][z + 8] - v_z[t1][x + 8][y + 10][z + 8]))*(mu[x + 8][y + 8][z + 8] + mu[x + 8][y + 8][z + 9] + mu[x + 8][y + 9][z + 8] + mu[x + 8][y + 9][z + 9])*damp[x + 1][y + 1][z + 1] + damp[x + 1][y + 1][z + 1]*tau_yz[t0][x + 8][y + 8][z + 8];
              tau_zz[t1][x + 8][y + 8][z + 8] = r40 + 2.804F*(6.97544653e-5F*(r43 + v_z[t1][x + 8][y + 8][z + 4]) + 1.19628908e-1F*(r46 + v_z[t1][x + 8][y + 8][z + 8]) + 9.57031264e-4F*(r49 + v_z[t1][x + 8][y + 8][z + 10]) + 7.97526054e-3F*(r52 + v_z[t1][x + 8][y + 8][z + 6]))*damp[x + 1][y + 1][z + 1]*mu[x + 8][y + 8][z + 8] + damp[x + 1][y + 1][z + 1]*tau_zz[t0][x + 8][y + 8][z + 8];
            }
          }
        }
      }
    }
  }
}
