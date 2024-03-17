#include "mdl.cuh"

#include "../../impl_tmpl/tmpl_etc.cu"

float mdl_les_gains(Mdl_t * mdl, uint t0, uint t1, uint mode, float GRAND_COEF, uint _t_MODE, uint GRAINE) {
	ASSERT(GRAND_COEF >= 2);
	ASSERT(mdl->T == (t1-t0));
	float * _y = gpu_vers_cpu<float>(mdl->y__d[C-1], (t1-t0)*P);
	float somme = 0;
	float potentiel = 0;
	FOR(t0, t, t1) {
		uint depart_plus_t = t_MODE_GENERALE(_t_MODE, GRAINE, t0, DEPART, FIN, (t-t0));
		//
		somme     += powf(fabs(prixs[/*t*/depart_plus_t+1]/prixs[/*t*/depart_plus_t]-1),GRAND_COEF) * (signe((prixs[/*t*/depart_plus_t+1]/prixs[/*t*/depart_plus_t]-1)) == signe(_y[(t-t0)*P+0]));
		potentiel += powf(fabs(prixs[/*t*/depart_plus_t+1]/prixs[/*t*/depart_plus_t]-1),GRAND_COEF);
	}
	free(_y);
	return somme / potentiel;
};

float mdl_score(Mdl_t * mdl, uint t0, uint t1, uint mode, uint _t_MODE, uint GRAINE) {
	ASSERT(mdl->T == (t1-t0));
	if (mode == 0) mdl_zero_cpu(mdl);
	else           mdl_zero_gpu(mdl);
	//
	mdl_f(mdl, t0, t1, mode, _t_MODE, GRAINE);
	//
	float somme_score;
	if (mode == 0) somme_score =  intel_somme_score(mdl->y[C-1],    t0, (t1-t0), _t_MODE, GRAINE);
	else           somme_score = nvidia_somme_score(mdl->y__d[C-1], t0, (t1-t0), _t_MODE, GRAINE);
	//
	if (mode == 0) return  intel_score_finale(somme_score, (t1-t0), _t_MODE, GRAINE);
	else           return nvidia_score_finale(somme_score, (t1-t0), _t_MODE, GRAINE);
};

float* mdl_pred(Mdl_t * mdl, uint t0, uint t1, uint mode, uint _t_MODE, uint GRAINE) {
	ASSERT(mdl->T == (t1-t0));
	if (mode == 0) mdl_zero_cpu(mdl);
	else           mdl_zero_gpu(mdl);
	//
	mdl_f(mdl, t0, t1, mode, _t_MODE, GRAINE);
	if (mode == 0) return  intel_prediction(mdl->y[C-1], t0, (t1-t0), _t_MODE, GRAINE);
	else           return nvidia_prediction(mdl->y__d[C-1], t0, (t1-t0), _t_MODE, GRAINE);
};

void mdl_aller_retour(Mdl_t * mdl, uint t0, uint t1, uint mode, uint _t_MODE, uint GRAINE) {
	ASSERT(mdl->T == (t1-t0));
	if (mode == 0) mdl_zero_cpu(mdl);
	else           mdl_zero_gpu(mdl);
	mdl_f(mdl, t0, t1, mode, _t_MODE, GRAINE);
	//
	float somme_score;
	if (mode == 0) somme_score =  intel_somme_score(mdl->y[C-1], t0, (t1-t0), _t_MODE, GRAINE);
	else           somme_score = nvidia_somme_score(mdl->y__d[C-1], t0, (t1-t0), _t_MODE, GRAINE);
	//
	float d_score;
	if (mode == 0) d_score =  d_intel_score_finale(somme_score, (t1-t0), _t_MODE, GRAINE);
	else           d_score = d_nvidia_score_finale(somme_score, (t1-t0), _t_MODE, GRAINE);
	//
	if (mode == 0)  d_intel_somme_score(d_score, mdl->y[C-1],    mdl->dy[C-1], t0, (t1-t0), _t_MODE, GRAINE);
	else           d_nvidia_somme_score(d_score, mdl->y__d[C-1], mdl->dy__d[C-1], t0, (t1-t0), _t_MODE, GRAINE);
	mdl_df(mdl, t0, t1, mode, _t_MODE, GRAINE);
};