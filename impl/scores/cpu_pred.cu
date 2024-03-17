#include "S.cuh"

#include "../../impl_tmpl/tmpl_etc.cu"

float* intel_prediction(float * y, uint depart, uint T, uint _t_MODE, uint GRAINE) {
	float * pourcent = (float*)calloc(P, sizeof(float));
	//
	FOR(0, i, T) {
		uint depart_plus_t = t_MODE_GENERALE(_t_MODE, GRAINE, depart, DEPART, FIN, i);
		FOR(0, p, P) {
			if (signe(y[(0+i)*P+p]) == signe(prixs[/*(depart+i)*/depart_plus_t+p+1]/prixs[/*depart+i*/depart_plus_t/*+p*/]-1)) {
				pourcent[p] += 1.0;
			}
		}
	}
	//
	FOR(0, p, P)
		pourcent[p] /= (float)T;
	//
	return pourcent;
};