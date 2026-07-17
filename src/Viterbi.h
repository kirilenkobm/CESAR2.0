/**
 * Viterbi function declaration.
 * Copyright 2017 MPI-CBG/MPI-PKS Peter Schwede
 */

//#include "Literal.h"
#include "State.h"
#include "HMM.h"
#include "Params.h"

struct LogoddMatrix* Viterbi__init_logodd_matrix(struct HMM* hmm);

struct PathMatrix* Viterbi__init_path_matrix(struct HMM* hmm, size_t num_observations);

LOGODD_T Viterbi__get_emission_logodd(Literal* observations, size_t t, struct State* state);

size_t Viterbi__checkpoint_block_size(size_t num_observations);

size_t Viterbi__estimate_dense_traceback_bytes(size_t num_states, size_t num_observations);

size_t Viterbi__estimate_checkpoint_traceback_bytes(size_t num_states, size_t num_observations);

void Viterbi(struct HMM* hmm, size_t num_observations, Literal* observation,
    size_t* path_length, struct State** path, TracebackMode traceback_mode);
