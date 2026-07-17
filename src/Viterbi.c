/**
 * Perform the viterbi on a given HMM.
 * Copyright 2017 MPI-CBG/MPI-PKS Peter Schwede
 */

// http://www.geeksforgeeks.org/dynamically-allocate-2d-array-c/

#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>

#include "Logging.h"
#include "Stateid.h"
#include "EmissionTable.h"
#include "Transition.h"
#include "Matrix.h"
#include "SafeAlloc.h"

#include "Viterbi.h"

typedef struct ViterbiCheckpoints {
  size_t block_size;
  size_t num_checkpoints;
  size_t num_states;
  LOGODD_T* scores;
} ViterbiCheckpoints;

size_t Viterbi__checkpoint_block_size(size_t num_observations) {
  if (num_observations == 0) {
    return 1;
  }

  // Balance four LOGODD_T score columns per checkpoint against one packed
  // traceback block (two path entries per byte).
  size_t block_size = (size_t) ceil(sqrt(8.0 * sizeof(LOGODD_T) * num_observations));
  return block_size > num_observations ? num_observations : block_size;
}

size_t Viterbi__estimate_dense_traceback_bytes(size_t num_states, size_t num_observations) {
  return ((num_observations + 1) * num_states + 1) / 2;
}

size_t Viterbi__estimate_checkpoint_traceback_bytes(size_t num_states, size_t num_observations) {
  if (num_states == 0 || num_observations == 0) {
    return 0;
  }

  const size_t block_size = Viterbi__checkpoint_block_size(num_observations);
  const size_t num_checkpoints = (num_observations - 1) / block_size + 1;
  const size_t checkpoint_bytes = num_checkpoints * 4 * num_states * sizeof(LOGODD_T);
  const size_t block_path_bytes = ((block_size + 1) * num_states + 1) / 2;
  const size_t start_path_bytes = (num_states + 1) / 2;
  return checkpoint_bytes + block_path_bytes + start_path_bytes;
}

static struct ViterbiCheckpoints* Viterbi__create_checkpoints(
    size_t num_states, size_t num_observations) {
  struct ViterbiCheckpoints* self = SAFEMALLOC(sizeof(struct ViterbiCheckpoints));
  self->block_size = Viterbi__checkpoint_block_size(num_observations);
  self->num_checkpoints = (num_observations - 1) / self->block_size + 1;
  self->num_states = num_states;
  self->scores = SAFEMALLOC(
      self->num_checkpoints * 4 * self->num_states * sizeof(LOGODD_T));
  return self;
}

static void Viterbi__destroy_checkpoints(struct ViterbiCheckpoints* self) {
  free(self->scores);
  free(self);
}

static void Viterbi__save_checkpoint(
    struct ViterbiCheckpoints* self, size_t t, struct LogoddMatrix* vmatrix) {
  size_t checkpoint = t / self->block_size;
  memcpy(&self->scores[checkpoint * 4 * self->num_states], vmatrix->v,
      4 * self->num_states * sizeof(LOGODD_T));
}

static void Viterbi__restore_checkpoint(
    struct ViterbiCheckpoints* self, size_t t, struct LogoddMatrix* vmatrix) {
  size_t checkpoint = t / self->block_size;
  memcpy(vmatrix->v, &self->scores[checkpoint * 4 * self->num_states],
      4 * self->num_states * sizeof(LOGODD_T));
}

/**
 * Create a new viterbi matrix and fill its values with -inf
 * @param hmm pointer to hidden markov model
 * @param num_observations query length
 */
struct LogoddMatrix* Viterbi__init_logodd_matrix(struct HMM* hmm) {
	LOGODD_T default_value = LOGODD_NEGINF;

  struct LogoddMatrix* vmatrix = LogoddMatrix__create(4, hmm->num_states, default_value);  // 4 = 3 previous + 1 current

  return vmatrix;
}

/**
 * Create a new path matrix and fill its values with empty state ids.
 * STATE_MAX_ID is used as empty state id.
 * @param hmm pointer to hidden markov model
 * @param num_observations query length
 */
struct PathMatrix* Viterbi__init_path_matrix(struct HMM* hmm, size_t num_observations) {
  struct PathMatrix* pmatrix = PathMatrix__create(num_observations + 1, hmm->num_states, PATH_EMPTY);  // +1 for virtual start state

  // Start states have no chosen incoming transition; PATH_EMPTY marks them as
  // terminal for the backtrace (recovered as a self-reference), matching the
  // former "predecessor == self" convention.
  for (uint16_t i = 0; i < hmm->num_starts; i++) {
    struct Transition start = hmm->starts[i];
    PathMatrix__set(pmatrix, 0, start.origin, PATH_EMPTY);
  }

  return pmatrix;
}


/**
 * Look up the emission probability of a state during an observation.
 * @param observations the querry.
 * @param t the point of time of observation -- an index for observations.
 * @param state the State whom the caller requests the emission of
 */
LOGODD_T Viterbi__get_emission_logodd(Literal* observations, size_t t, struct State* state) {
  if (state->num_emissions == 0) {
    return 0;
  }
  if (t < state->num_emissions) {
    return LOGODD_NEGINF;
  }
  for (uint16_t i=0; i < state->num_emissions; i++) {
    logv(6, "Observation: %lu", t);
    logv(6, "Logodd lookup for: %c [%u]", Literal__char(observations[t-state->num_emissions]), i);
  }

  return EmissionTable__by_literals(
      state->emission_table,
      state->reference,
      &observations[t-state->num_emissions]
      );
}


/**
 * Perform the viterbi recursion step at time t.
 * @param vmatrix the viterbi matrix
 * @param pmatrix the path matrix
 * @param hmm the hidden markov model
 * @param observations the sequence of observed literals
 * @param t is assumed to be greater than zero (past the initialization).
 *
 * # Viterbi sub steps at observation i for an HMM with silent states and
 *   without silent loops: (Durbin et al., 1998, p. 71)
 *   (i) For all emitting states l, calculate viterbi(l, i) from
 *       max(viterbi(k, i-x)+p)+e where p is the prob of t(l,p,k)
 *  (ii) For all silent states l, set viterbi(l, i) to max(viterbi(k, i))*p
 *       where p is the probability of t=(l,p,k) (no e because silent)
 * (iii) Starting from the lowest numbered silent state l set viterbi(l, i) to
 *       max(viterbi(k, i)+p) for all silent states k < l
 *
 * Basically first serve all emitting states, then all silent states and
 * finally let silent states driple down their probs before starting with
 * emitting states again.
 *
 * Those steps can be reduced to one general step only iff there is no doubt
 * that all silent chains from states A to B in a lower probability compared to
 * any emitting chain from A to B.
 *
 * # Each a viterbi sub step:
 * 1. for each state k1 at time t
 * 1.1 for each transition a=(k0,p,k1) to k1 select the maximum probability pmax=(viterbi(t-1,k0)*p)
 * 1.2 multiply pm with emission probability of observation at time t
 * 1.3 assign viterbi(t,k1) := pmax
 * 1.4 assign path(t-num_emissions, k1) := k0
 */
void Viterbi__step(struct LogoddMatrix* vmatrix, struct PathMatrix* pmatrix, struct HMM* hmm, Literal* observations, size_t t, size_t path_t) {
  if (t != 0) {
    for(STATE_ID_T sid=0; sid < hmm->num_states; sid++) {
      LogoddMatrix__set(vmatrix, t%4, sid, LOGODD_NEGINF);
    }
  }

  uint16_t silent = 0;
  if (t==0) {
    silent = 2;
  }

  for(; silent<=2; silent++) {

    // 1.
    for (STATE_ID_T i = 0; i < hmm->num_states; i++) {
      struct State* state = &hmm->states[i];

      LOGODD_T max_logodd = LOGODD_NEGINF;
      STATE_ID_T origin_id = STATE_MAX_ID;
      PATH_ENTRY_T origin_index = PATH_EMPTY;  // index into state->incoming of the winning edge

      if (state->num_emissions > t) {
        logv(6, "t=%lu\ti=%s="SID"\tCurrent state emits too much to emit this early:\t%u > %lu", t, state->name, i, state->num_emissions, t);
        continue;
      }

      if (silent == 0 && state->num_emissions == 0) {
        continue;
      }
      if (silent > 0 && state->num_emissions > 0) {
        continue;
      }

      LOGODD_T emission_logodd = Viterbi__get_emission_logodd(observations, t, state);
      
      char ref[4] = "", qry[4] = "";
      if (g_loglevel > 5) {
        Literal__str(state->num_emissions, state->reference, ref);
        Literal__str(state->num_emissions, &observations[t-state->num_emissions], qry);
        logv(6, "t=%lu\ti=%s="SID"\tqry=%s=%i\tref=%s=%i\temission_logodd=%E",
            t, state->name, i,
            qry, Literal__uint(state->num_emissions, &observations[t-state->num_emissions]),
            ref, Literal__uint(state->num_emissions, state->reference),
            emission_logodd);
      }

      if (state->num_emissions > 0 && emission_logodd == LOGODD_NEGINF) {
        if (g_loglevel > 5) {
          logv(6, "t=%lu\ti=%s="SID"\tCurrent state cannot emit observation\t%s:\t%E", t, state->name, i, qry, emission_logodd);
        }
        continue;
      }

      // 1.1 for each transition t=(k0, p, k1) select the maximum probability pmax=(viterbi(t-1,k0)*p)
      logv(6, "t=%lu\ti=%s="SID"\t1 select max prob", t, state->name, i);
      for (size_t j=0; j < state->num_incoming; j++) {
        struct Transition transition = state->incoming[j];

        if (silent == 2) {
          if (transition.origin >= i) {
            continue;
          }
          if (hmm->states[transition.origin].num_emissions > 0) {
            continue;
          }
        }

        LOGODD_T origin_logodd = LogoddMatrix__get(vmatrix, ((size_t) t-state->num_emissions)%4, transition.origin);

        logv(6, "t=%lu\ti=%s="SID"\tcheck (%lu / %u), t'=t-%u", t, state->name, i, j+1, state->num_incoming, state->num_emissions);
        logv(6, "t=%lu\ti=%s="SID"\torigin_logodd=%E\tt.logodd=%E\t\tt.origin=%s="SID, t, state->name, i, origin_logodd, transition.logodd, hmm->states[transition.origin].name, transition.origin);

        if (origin_logodd == LOGODD_NEGINF) {
          continue;
        }

        // pmax = viterbi(t-1,k0)*p
        LOGODD_T sum = Logodd__add(transition.logodd, origin_logodd);

        // is pmax maximal?
        if (max_logodd < sum) {
          max_logodd = sum;
          origin_id = transition.origin;
          origin_index = (PATH_ENTRY_T) j;  // remember which incoming edge won
          logv(6, "t=%lu\ti=%s="SID"\tnew max_logodd:\t%E\t"SID, t, state->name, i, max_logodd, origin_id);
        }

      }  // O( state->incoming ) = O( deg+(k) )


      if (max_logodd == LOGODD_NEGINF) {
        logv(6, "t=%lu\ti=%s="SID"\tmax_logodd=%E\torigin=%s="SID"", t, state->name, i, max_logodd, "[]", origin_id);
        continue;
      }
      logv(6, "t=%lu\ti=%s="SID"\tmax_logodd=%E\torigin=%s="SID"", t, state->name, i, max_logodd, hmm->states[origin_id].name, origin_id);


      // 1.2 multiply pm with emission probability of observation at time t
      logv(6, "t=%lu\ti=%s="SID"\t2 multiply emission logodd %E", t, state->name, i, emission_logodd);
      max_logodd = Logodd__add(emission_logodd, max_logodd);

      // 1.3 assign viterbi(t,k1) <= pm
      if (max_logodd > LogoddMatrix__get(vmatrix, ((size_t) t)%4, state->id)) {
        LOGODD_T prevv = LogoddMatrix__get(vmatrix, t%4, state->id);
        if (prevv == LOGODD_NEGINF) {
          logv(6, "t=%lu\ti=%s="SID"\tassign v(%lu,%s="SID") = -inf := %E", t, state->name, i, t, state->name, state->id, max_logodd);
        } else {
          logv(6, "t=%lu\ti=%s="SID"\tassign v(%lu,%s="SID") = %E := %E", t, state->name, i, t, state->name, state->id, prevv, max_logodd);
        }
        LogoddMatrix__set(vmatrix, t%4, state->id, max_logodd);
        logv(6, "t=%lu\ti=%s="SID"\tassign p(%lu,%s="SID") := "SID" (incoming #%u)", t, state->name, i, t, state->name, state->id, origin_id, (unsigned) origin_index);
        if (pmatrix != NULL) {
          PathMatrix__set(pmatrix, path_t, state->id, origin_index);
        }
      }
    }
  }  // hmm->states  O( deg+(k) * S + s*s) (s = silent states in S)
}  // O(S * deg+(k))

/**
 * Run viterbi on an HMM.
 * The given path pointer contains the most probable path through the HMM that ends in an ending state.
 * Dense mode retains the packed S*T path matrix. Checkpoint mode saves score
 * snapshots and recomputes short path blocks on demand.
 * @param hmm the HMM.
 * @param num_observations the length of the query.
 * @param observations the array of observed literals.
 * @param path_length a maximum length of the resulting path.
 * @param path a pointer to an array for the sequence of state pointers.
 */
void Viterbi(struct HMM* hmm, size_t num_observations, Literal* observations,
    size_t* path_length, struct State** path, TracebackMode traceback_mode) {
  if (num_observations == 0) {
    die("No observations.");
  }

  // Running viterbi on an HMM without any states will return an empty list of states.
  // The same is the case if the HMM has no start or end states.
  if (hmm->num_states == 0) {
    die("HMM is empty.");
  }
  if (hmm->num_starts == 0 ) {
    die("HMM has zero start states.");
  }
  if (hmm->num_ends == 0) {
    die("HMM has zero end states.");
  }

  logv(1, "Num states:\t%lu", hmm->num_states);
  logv(1, "Num observations:\t%lu", num_observations);

  // The init of a dynamically sized 2d array requires >=c99
  struct LogoddMatrix* vmatrix = Viterbi__init_logodd_matrix(hmm);
  struct PathMatrix* dense_path = NULL;
  struct PathMatrix* start_path = NULL;
  struct ViterbiCheckpoints* checkpoints = NULL;
  if (traceback_mode == TRACEBACK_DENSE) {
    dense_path = Viterbi__init_path_matrix(hmm, num_observations);
  } else {
    start_path = PathMatrix__create(1, hmm->num_states, PATH_EMPTY);
    checkpoints = Viterbi__create_checkpoints(hmm->num_states, num_observations);
  }

  // init the first column: the initial states.
  for (size_t i = 0; i < hmm->num_starts; i++) {
    struct Transition transition = hmm->starts[i];
    LogoddMatrix__set(vmatrix, 0, transition.origin, transition.logodd);
  }  // O(S)

  for (size_t t=0; t <= num_observations; t++) {
    struct PathMatrix* forward_path = traceback_mode == TRACEBACK_DENSE
        ? dense_path
        : (t == 0 ? start_path : NULL);
    Viterbi__step(vmatrix, forward_path, hmm, observations, t, t);
    if (traceback_mode == TRACEBACK_CHECKPOINT &&
        t < num_observations && t % checkpoints->block_size == 0) {
      Viterbi__save_checkpoint(checkpoints, t, vmatrix);
    }
  }  // O(deg+(k) * S * T) = O(S * T)

  if (g_loglevel > 3) {
    if (hmm->num_states < 200 && num_observations < 300) {
      char tmp[1024000] = "";
      FILE * matrixlog = fopen("cesar_matrix.log", "w");
      fprintf(matrixlog, "###states\n");
      for (size_t i=0; i < hmm->num_states; i++) {
        State__str(&hmm->states[i], tmp);
        fprintf(matrixlog, "%lu\t%s\n", i, tmp);
      }
      fprintf(matrixlog, "###states_end\n");
      LogoddMatrix__str(vmatrix, tmp);
      fprintf(matrixlog, "###vmatrix:\n%s\n###vmatrix_end\n", tmp);
      fclose(matrixlog);
    }
  }

  /**
   * Find the backtrace start
   */
  LOGODD_T logodd = LOGODD_NEGINF;
  STATE_ID_T best_end = STATE_MAX_ID;
  for (size_t k=0; k < hmm->num_ends; k++) {
    Transition end = hmm->ends[k];
    LOGODD_T current = LogoddMatrix__get(vmatrix, (size_t) num_observations%4, (size_t) end.origin);
    LOGODD_T sum = Logodd__add(current, end.logodd);
    logv(2, "Checking end:\t%s\t%lE", hmm->states[end.origin].name, sum);
    if (logodd < sum) {
      best_end = end.origin;
      logodd = sum;
    }
  }  // O(S)

  logv(1, "LogoddMatrix (WxH):\t%lu x %lu", vmatrix->num_columns, vmatrix->num_rows);
  logv(1, "LogoddMatrix (bytes):\t%lu", LogoddMatrix__bytes(vmatrix));
  if (traceback_mode == TRACEBACK_DENSE) {
    logv(1, "Traceback mode:\tdense");
    logv(1, "Traceback storage (bytes):\t%lu", PathMatrix__bytes(dense_path));
  } else {
    logv(1, "Traceback mode:\tcheckpoint");
    logv(1, "Traceback checkpoint block size:\t%lu", checkpoints->block_size);
    logv(1, "Traceback checkpoints:\t%lu", checkpoints->num_checkpoints);
    logv(1, "Traceback storage (bytes):\t%lu",
        Viterbi__estimate_checkpoint_traceback_bytes(hmm->num_states, num_observations));
  }
  logv(1, "States (bytes):\t%lu", sizeof(State) * hmm->num_states);

  if (best_end == STATE_MAX_ID || logodd == LOGODD_NEGINF) {
    die("No valid path found.");
  }


  logv(1, "Viterbi logodd:\t%lE", logodd);
  logv(1, "Viterbi prob:\t%lE", Logodd__exp(logodd));

  *path_length = num_observations*2;

  logv(4, "path[%lu] := "SID, *path_length-1, best_end);
  path[*path_length-1] = &hmm->states[best_end];

  size_t i = *path_length - 1;
  size_t t = num_observations;
  if (traceback_mode == TRACEBACK_DENSE) {
    for (; i > 0; i--) {
      struct State* state = path[i];
      PATH_ENTRY_T incoming_index = PathMatrix__get(dense_path, t, state->id);
      STATE_ID_T predecessor = (incoming_index == PATH_EMPTY)
          ? state->id
          : state->incoming[incoming_index].origin;
      path[i-1] = &hmm->states[predecessor];
      if (t == 0 && state->num_emissions > 0) {
        break;
      }
      t -= state->num_emissions;
    }
  } else while (i > 0 && t > 0) {
    // Always choose the checkpoint strictly before the current traceback
    // coordinate, so the endpoint's backpointer is recomputed in this block.
    const size_t block_start = ((t - 1) / checkpoints->block_size) * checkpoints->block_size;
    const size_t block_columns = t - block_start + 1;
    struct PathMatrix* block_path =
        PathMatrix__create(block_columns, hmm->num_states, PATH_EMPTY);

    Viterbi__restore_checkpoint(checkpoints, block_start, vmatrix);
    for (size_t recompute_t = block_start + 1; recompute_t <= t; recompute_t++) {
      Viterbi__step(vmatrix, block_path, hmm, observations, recompute_t,
          recompute_t - block_start);
    }

    while (i > 0 && t > block_start) {
      struct State* state = path[i];
      PATH_ENTRY_T incoming_index =
          PathMatrix__get(block_path, t - block_start, state->id);
      STATE_ID_T predecessor = (incoming_index == PATH_EMPTY)
          ? state->id
          : state->incoming[incoming_index].origin;
      path[i-1] = &hmm->states[predecessor];

      if (g_loglevel >= 4) {
        char qry[4] = "", ref[4] = "";
        Literal__str(state->num_emissions, &observations[t-state->num_emissions], qry);
        Literal__str(state->num_emissions, state->reference, ref);
      /*
      uint16_t qry_id = Literal__uint(state->num_emissions, &observations[t-state->num_emissions]);
      uint16_t ref_id = Literal__uint(state->num_emissions, state->reference);
      logv(4,
          "path[%lu] := Pget(%lu, %s="SID") = %s="SID"(%u)\te(qry=%s=%i, ref=%s=%i)=%E\tv=%E",
          i-1,
          t, state->name, state->id,
          path[i-1]->name, path[i-1]->id, path[i-1]->num_emissions,
          qry, qry_id, ref, ref_id, Viterbi__get_emission_logodd(observations, t, state),
          LogoddMatrix__get(vmatrix, t, state->id));
      */
      }

      t -= state->num_emissions;
      i--;
    }
    PathMatrix__destroy(block_path);
  }

  while (traceback_mode == TRACEBACK_CHECKPOINT && i > 0) {
    struct State* state = path[i];
    PATH_ENTRY_T incoming_index = PathMatrix__get(start_path, 0, state->id);
    STATE_ID_T predecessor = (incoming_index == PATH_EMPTY)
        ? state->id
        : state->incoming[incoming_index].origin;
    path[i-1] = &hmm->states[predecessor];
    if (state->num_emissions > 0) {
      break;
    }
    i--;
  }

  // shrink path
  const size_t offset = i;
  for(i=0; i < *path_length - offset; i++) {
    if (i < *path_length - offset) {
      path[i] = path[i+offset];
    } else {
      path[i] = NULL;
    }
  }  // O(T)
  *path_length -= offset;

  LogoddMatrix__destroy(vmatrix);
  if (traceback_mode == TRACEBACK_DENSE) {
    PathMatrix__destroy(dense_path);
  } else {
    PathMatrix__destroy(start_path);
    Viterbi__destroy_checkpoints(checkpoints);
  }
}
