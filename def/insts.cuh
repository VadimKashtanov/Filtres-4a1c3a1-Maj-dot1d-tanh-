#pragma once

#include "etc.cuh"

/*
Deux options d'optimisation s'offrent a moi
	1) depart -> depart + T                             Soit on fait une suite continue de t
	2) [randint(depart, fin) for _ in range(T)]         Soit des t aleatoirement choisis parmis DEPART:FIN
*/

//	Si MODE==t_CONTINUE alors    t = depart+i
//	Si MODE==t_PSEUDO_ALEA alors t = pseudo(grain, i)

#define t_CONTINUE        0 //t0+0,t0+1,t0+2,...
#define t_PSEUDO_ALEA     1 //rnd(),rnd(),rnd()...
#define t_PSEUDO_ALEA_x16 2 //[[a+i for i in range(16)] for a=rnd()]

#define t_MODE(MODE, _depart, _fin, i, graine) ( \
	MODE == t_CONTINUE        ? (_depart+i)                                         : \
	MODE == t_PSEUDO_ALEA     ? (_depart+PSEUDO_ALEA((graine+i)) % (_fin-_depart-1) ) : \
	MODE == t_PSEUDO_ALEA_x16 ? (_depart+(PSEUDO_ALEA((i-i%16)/16) % (_fin-_depart-16-1)) + (i-i%16)) : \
	NULL)

#define t_MODE_GENERALE(MODE, GRAINE, _depart, _DEPART, _FIN, i) ( \
	MODE == t_CONTINUE        ? t_MODE(MODE, _depart, 0, i, 0)         : \
	MODE == t_PSEUDO_ALEA     ? t_MODE(MODE, _DEPART, _FIN, i, GRAINE) : \
	MODE == t_PSEUDO_ALEA_x16 ? t_MODE(MODE, _DEPART, _FIN, i, GRAINE) : \
	NULL)

/*	Ne seront munis de pseudo-alea seulement le Dar et l'Exi. Car le reste se refere au GRAND_T et n'est pas liée au temps.

Seule les normalisee_diff[] auront le _t_MODE

et donc le SCORE(x) et dSCORE(c)

*/