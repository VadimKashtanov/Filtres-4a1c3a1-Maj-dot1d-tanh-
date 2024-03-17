#include "filtres_prixs_tolerant.cuh"

#define BLOQUE_T  4//8
#define BLOQUE_B  4//8
#define BLOQUE_FB 8//16//32//8

#include "../../../impl_tmpl/tmpl_etc.cu"

static __device__ float filtre_device(float * x, float * dif_x, float * f, float * locd) {
	float s = 0;
	float f_nouveau = f[0];
	s += powf(1 + fabs(x[0] - f_nouveau), 1.0/*(0.5+0/N*0.5)*/);
	FOR(1, i, N) {
		f_nouveau = f[i];
		float Ps = 1.0;//(0.5+i/N*0.5);
		//s += powf(1 + fabs(   x[i]  -       f_nouveau    ), 0.5);
		s += powf(1 + fabs(   x[i]  -       f_nouveau    ), Ps);
	};

	s = s/(float)N-1;
	
	float y = expf(-s*s);

	locd[0] = -2*2*s*y;

	return 2*y-1;
	//return 2*filtres_f_info(y)-1;
};

static __global__ void kerd_filtre_naive(
	uint _t_MODE, uint GRAINE,
	uint X_vars, uint Y_vars,
	uint depart, uint T,
	uint bloques, uint f_par_bloque,
	float * x, float * dif_x,
	float * f,
	float * y,
	float * locd)
{
	uint _t = threadIdx.x + blockIdx.x * blockDim.x;
	uint _b = threadIdx.y + blockIdx.y * blockDim.y;
	uint _f = threadIdx.z + blockIdx.z * blockDim.z;

	if (_t < T && _b < bloques && _f < f_par_bloque) {
		uint cuda_depart_plus_t = t_MODE_GENERALE(_t_MODE, GRAINE, depart, DEPART, FIN, _t);
		y[(0+_t)*(bloques*f_par_bloque) + _b*f_par_bloque + _f] = filtre_device(
			x     + _b*PRIXS*N_FLTR + /*(depart+_t)*/cuda_depart_plus_t*N_FLTR,
			dif_x + _b*PRIXS*N_FLTR + /*(depart+_t)*/cuda_depart_plus_t*N_FLTR,
			f     + _b*f_par_bloque*N     + _f*N,
			locd  + (0+_t)*bloques*f_par_bloque*1 + _b*f_par_bloque*1 + _f*1
		);
	}
};

void nvidia_filtres_prixs_tolerant___naive(
	uint _t_MODE, uint GRAINE,
	uint X_vars, uint Y_vars,
	uint depart, uint T,
	uint bloques, uint f_par_bloque,
	float * x, float * dif_x,
	float * f,
	float * y,
	float * locd)
{
	kerd_filtre_naive<<<dim3(KERD(T, BLOQUE_T), KERD(bloques, BLOQUE_B), KERD(f_par_bloque, BLOQUE_FB)), dim3(BLOQUE_T, BLOQUE_B, BLOQUE_FB)>>>(
		_t_MODE, GRAINE,
		X_vars, Y_vars,
		depart, T,
		bloques, f_par_bloque,
		x, dif_x,
		f,
		y,
		locd);
	ATTENDRE_CUDA();
}

__device__ static void d_nvidia_filtre(float * x, float * dif_x, float * f, float * locd, float * dy, float * df) {
	/*float ds = locd[0] * dy[0] / 8;
	float dd = locd[1] * dy[0] / 7;
	//
	FOR(1, i, N)
	{
		//s += sqrtf(1 + fabs(  x[i]   -   f[i]  ));
		atomicAdd(&df[i], ds * 1 / (2*sqrtf(1 + fabs(x[i] - f[i]))) * (-1) * cuda_signe(x[i] - f[i]));
		//d += powf((1 + fabs(dif_x[i] - dif_f[i])), 2);
		atomicAdd(&df[ i ], dd * 2 * (1 + fabs(dif_x[i] - (f[i]-f[i-1]))) * cuda_signe(dif_x[i] - (f[i]-f[i-1])) * (-1));
		atomicAdd(&df[i-1], dd * 2 * (1 + fabs(dif_x[i] - (f[i]-f[i-1]))) * cuda_signe(dif_x[i] - (f[i]-f[i-1])) * (+1));
	}
	atomicAdd(&df[0], ds * 1 / (2*sqrtf(1 + fabs(x[0] - f[0]))) * (-1) * cuda_signe(x[0] - f[0]));*/

	float ds = locd[0] * dy[0] / 8;
	//
	FOR(1, i, N)
	{
		float Ps = 1.0;//(0.5+i/N*0.5);
		
		//s += powf(1 + fabs(   x[i]  -       f_nouveau    ), (0.5+i/N*0.5));
		atomicAdd(&df[i], ds * Ps * powf(1 + fabs(x[i] - f[i]), Ps-1) * (-1) * cuda_signe(x[i] - f[i]));
	}
	float Ps = 1.0;//(0.5+0/N*0.5);
	//df[0] += ds * 1 / (2*sqrtf(1 + fabs(x[0] - f[0]))) * (-1) * signe(x[0] - f[0]);
	atomicAdd(&df[0], ds * Ps * powf(1 + fabs(x[0] - f[0]), Ps-1) * (-1) * cuda_signe(x[0] - f[0]));

};

__global__ static void  d_nvidia_kerd_filtre_naive(
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
	uint _t = threadIdx.x + blockIdx.x * blockDim.x;
	uint _b = threadIdx.y + blockIdx.y * blockDim.y;
	uint _f = threadIdx.z + blockIdx.z * blockDim.z;

	if (_t < T && _b < bloques && _f < f_par_bloque) {
		uint cuda_depart_plus_t = t_MODE_GENERALE(_t_MODE, GRAINE, depart, DEPART, FIN, _t);
		d_nvidia_filtre(
				x + _b*PRIXS*N_FLTR + /*(depart+_t)*/cuda_depart_plus_t*N_FLTR,
			dif_x + _b*PRIXS*N_FLTR + /*(depart+_t)*/cuda_depart_plus_t*N_FLTR,
			f     + _b*f_par_bloque*N     + _f*N,
			locd  + (0+_t)*(bloques*f_par_bloque*1) + _b*(f_par_bloque*1) + _f*1,
			dy    + (0+_t)*bloques*f_par_bloque + _b*f_par_bloque + _f,
			df    + _b*f_par_bloque*N     + _f*N
		);
	}
};

void d_nvidia_filtres_prixs_tolerant___naive(
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
	d_nvidia_kerd_filtre_naive<<<dim3(KERD(T, BLOQUE_T), KERD(bloques, BLOQUE_B), KERD(f_par_bloque, BLOQUE_FB)), dim3(BLOQUE_T, BLOQUE_B, BLOQUE_FB)>>>(
		_t_MODE, GRAINE,
		X_vars, Y_vars,
		depart, T,
		bloques, f_par_bloque,
		x, dif_x,
		f,
		y,
		locd,
		dy,
		df);
	ATTENDRE_CUDA();
}