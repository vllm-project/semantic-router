//go:build !windows && cgo && (amd64 || arm64 || riscv64)

package candle_binding

/*
#include <stdint.h>
extern char* candle_instance_load_pair_scorer(const char*, const char*);
extern char* candle_instance_score_pairs(uint64_t, const char*);
*/
import "C"

import "encoding/json"

func nativeInstanceLoadPairScorer(options InstanceOptions, selection PairScorerSelection) (uint64, error) {
	payload, err := json.Marshal(options)
	if err != nil {
		return 0, err
	}
	exit, err := json.Marshal(selection)
	if err != nil {
		return 0, err
	}
	args, free, err := instanceStrings(string(payload), string(exit))
	if err != nil {
		return 0, err
	}
	defer free()
	return decodeInstanceResult[uint64](C.candle_instance_load_pair_scorer(args[0], args[1]))
}

func nativeInstanceScorePairs(handle uint64, pairs []TextPair) (PairScores, error) {
	payload, err := json.Marshal(pairs)
	if err != nil {
		return PairScores{}, err
	}
	args, free, err := instanceStrings(string(payload))
	if err != nil {
		return PairScores{}, err
	}
	defer free()
	return decodeInstanceResult[PairScores](C.candle_instance_score_pairs(C.uint64_t(handle), args[0]))
}
