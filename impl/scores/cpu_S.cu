#include "S.cuh"

#include "../../impl_tmpl/tmpl_etc.cu"

float  intel_somme_score(float * y, uint depart, uint T, uint _t_MODE, uint GRAINE) {
	float somme = 0;
	FOR(0, t, T) {
		uint depart_plus_t = t_MODE_GENERALE(_t_MODE, GRAINE, depart, DEPART, FIN, t);
		FOR(0, p, P) {
			float alea = 0;
			y[(0+t)*P+p] += rnd()*SCORE_Y_COEF_BRUIT;
			somme += (P-p)*SCORE(y[(0+t)*P+p], prixs[/*depart+t*/depart_plus_t+p+1], prixs[/*depart+t*/depart_plus_t/*+p*/], alea);
		}
	}
	return somme;
};

float  intel_score_finale(float somme, uint T, uint _t_MODE, uint GRAINE) {
	return APRES_SCORE(somme / (float)(P * T));
};

//	=================================================

float d_intel_score_finale(float somme, uint T, uint _t_MODE, uint GRAINE) {
	return dAPRES_SCORE(somme / (float)(P * T)) / (float)(P * T);
};

void d_intel_somme_score(float d_somme, float * y, float * dy, uint depart, uint T, uint _t_MODE, uint GRAINE) {
	FOR(0, t, T) {
		uint depart_plus_t = t_MODE_GENERALE(_t_MODE, GRAINE, depart, DEPART, FIN, t);
		FOR(0, p, P) {
			float alea = 0;
			dy[(0+t)*P+p] = d_somme * (P-p)*dSCORE(y[(0+t)*P+p], prixs[/*depart+t*/depart_plus_t+p+1], prixs[/*depart+t*/depart_plus_t/*+p*/], alea) / (float)(T*P);
		}
	}
};