#include "filtres_prixs_tolerant.cuh"

#define BLOQUE_T  16

#define _repete_T 16

#include "../../../impl_tmpl/tmpl_etc.cu"

static __global__ void kerd_filtre_shared(
	uint _t_MODE, uint GRAINE,
	uint X_vars, uint Y_vars,
	uint depart, uint T,
	uint bloques,
	float * x, float * dif_x,
	float * f,
	float * y,
	float * locd)
{
	uint depart_bloque_t = blockIdx.y * BLOQUE_T * _repete_T;
	uint depart_thread_t = depart_bloque_t + threadIdx.y * _repete_T;

	uint _b = blockIdx.x;
	uint _f = blockIdx.z;	//(ligne dans bloque)

	uint LIGNE  = _b;
	uint BLOQUE = _b; 

	uint thx = threadIdx.x;
	uint thy__t = threadIdx.y;

	//if (_t < T)
	__shared__ float __f__[N];
	//
	if (thy__t==0) __f__[thx]  = f[BLOQUE*F_PAR_BLOQUES*N + _f*N + thx];
	__syncthreads();
	//
	float fi;
	fi = __f__[thx];
	//
	__shared__ float __ret[BLOQUE_T][1];	//s, d
	__shared__ float __y  [BLOQUE_T];
	//
	float xi;
	//
	uint _t;
	FOR(0, plus_t, _repete_T) {
		//
		_t = depart_thread_t + plus_t;

		uint cuda_depart_plus_t = t_MODE_GENERALE(_t_MODE, GRAINE, depart, DEPART, FIN, _t);
		//
		if (thx < 1) {
			__ret[thy__t][thx] = 0;
		}
		__syncthreads();
		//
		xi = x[LIGNE*PRIXS*N_FLTR + /*(depart+_t)*/cuda_depart_plus_t*N_FLTR + thx];
		//
		float Ps = 1.0;//(0.5+thx/N*0.5);
		atomicAdd(&__ret[thy__t][0], powf(1 + fabs(xi - fi), Ps));
		__syncthreads();
		//
		if (thx < 1) {
			__ret[thy__t][thx] = __ret[thy__t][thx]/(float)(8-thx) - 1.0;
		}
		__syncthreads();
		//
		if (thx < 1) {
			__y[thy__t] = expf(-__ret[thy__t][0]*__ret[thy__t][0]);
		}
		__syncthreads();
		//
		if (thx < 1) {
			locd[(0+_t)*BLOQUES*(F_PAR_BLOQUES*1) + BLOQUE*(F_PAR_BLOQUES*1) + _f*1 + thx] = -2*2*__ret[thy__t][thx]*__y[thy__t];
		}
		__syncthreads();
		//
		if (thx < 1) {
			y[(0+_t)*BLOQUES*F_PAR_BLOQUES + BLOQUE*F_PAR_BLOQUES + _f] = 2*__y[thy__t] - 1;
		}
	};
};

void nvidia_filtres_prixs_tolerant___shared(
	uint _t_MODE, uint GRAINE,
	uint X_vars, uint Y_vars,
	uint depart, uint T,
	uint bloques, uint f_par_bloque,
	float * x, float * dif_x,
	float * f,
	float * y,
	float * locd)
{
	ASSERT(BLOQUE_T*_repete_T <= T);
	kerd_filtre_shared<<<dim3(bloques, KERD((DIV(T,_repete_T)), BLOQUE_T), F_PAR_BLOQUES), dim3(N, BLOQUE_T,1)>>>(
		_t_MODE, GRAINE,
		X_vars, Y_vars,
		depart, T,
		bloques,
		x, dif_x,
		f,
		y,
		locd);
	ATTENDRE_CUDA();
};

static __global__ void d_kerd_filtre_shared(
	uint _t_MODE, uint GRAINE,
	uint X_vars, uint Y_vars,
	uint depart, uint T,
	uint bloques,
	float * x, float * dif_x,
	float * f,
	float * y,
	float * locd,
	float * dy,
	float * df)
{
	uint depart_bloque_t = blockIdx.y * BLOQUE_T * _repete_T;
	uint depart_thread_t = depart_bloque_t + threadIdx.y * _repete_T;

	uint _b = blockIdx.x;
	uint _f = blockIdx.z;	//(ligne dans bloque)

	uint LIGNE  = _b;
	uint BLOQUE = _b; 

	uint thx = threadIdx.x;
	uint thy__t = threadIdx.y;

	//if (_t < T)
	__shared__ float __f__[N];
	__shared__ float __df__[N];
	//
	if (thy__t==0) {
		__f__[thx]  = f[BLOQUE*F_PAR_BLOQUES*N + _f*N + thx];
		__df__[thx] = 0;
	}
	__syncthreads();
	//
	float fi;
	fi = __f__[thx];
	//
	__shared__ float __locd[BLOQUE_T][1];	//ds, dd
	__shared__ float __dy0[BLOQUE_T];
	//
	float xi;
	//
	uint _t;
	FOR(0, plus_t, _repete_T) {
		_t = depart_thread_t + plus_t;
		uint cuda_depart_plus_t = t_MODE_GENERALE(_t_MODE, GRAINE, depart, DEPART, FIN, _t);
		//
		if (thx < 1) {
			__dy0[thy__t] = dy[(0+_t)*BLOQUES*F_PAR_BLOQUES + BLOQUE*F_PAR_BLOQUES + _f];
		}
		__syncthreads();
		//
		if (thx < 1) {
			__locd[thy__t][thx] = locd[(0+_t)*BLOQUES*(F_PAR_BLOQUES*1) + BLOQUE*(F_PAR_BLOQUES*1) + _f*1 + thx] * __dy0[thy__t]/ (float)(8 - thx);
		}
		__syncthreads();
		//
		xi = x[LIGNE*PRIXS*N_FLTR + /*(depart+_t)*/cuda_depart_plus_t*N_FLTR + thx];
		//
		//atomicAdd(&__ret[thy__t][0], sqrtf(1 + fabs(xi - fi)));
		float Ps = 1.0;//(0.5+thx/N*0.5);
		atomicAdd(&__df__[thx], __locd[thy__t][0] * (Ps) * powf(1 + fabs(xi - fi), Ps-1) * (-1) * cuda_signe(xi - fi));
		__syncthreads();
	};
	__syncthreads();
	if (thy__t == 0) {
		atomicAdd(&df[BLOQUE*F_PAR_BLOQUES*N + _f*N + thx], __df__[thx]);
	}
};

void d_nvidia_filtres_prixs_tolerant___shared(
	uint _t_MODE, uint GRAINE,
	uint X_vars, uint Y_vars,
	uint depart, uint T,
	uint bloques, uint f_par_bloque,
	float * x, float * dif_x,
	float * f,
	float * y,
	float * locd,
	float * dy,
	float * df)
{
	ASSERT(BLOQUE_T*_repete_T <= T);
	d_kerd_filtre_shared<<<dim3(bloques, KERD((DIV(T,_repete_T)), BLOQUE_T), F_PAR_BLOQUES), dim3(N, BLOQUE_T,1)>>>(
		_t_MODE, GRAINE,
		X_vars, Y_vars,
		depart, T,
		bloques,
		x, dif_x,
		f,
		y,
		locd,
		dy,
		df);
	ATTENDRE_CUDA();
}