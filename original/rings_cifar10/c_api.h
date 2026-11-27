#pragma once
#include <cstddef>
#include <cstdint>
extern "C" {
void* rr_create(const char*,char*,std::size_t);
void rr_destroy(void*);
const char* rr_error(void*);
int rr_reset(void*);
std::size_t rr_feature_count(void*);
int rr_predict(void*,const std::uint8_t*,std::size_t,int,double*,int*);
int rr_predict_batch(void*,const std::uint8_t*,std::size_t,std::size_t,int,double*,int*);
// handle, frames, available bytes, frame count, byte stride, reset_each,
// scores, score capacity (elements), labels, label capacity (elements).
// Each frame is 32x32 RGB uint8; stride >= 3072. reset_each is 0 or 1.
// Bounds validated before writing. Zero frames permit null buffers.
// Returns 0 on success, -1 on failure; rr_error gives a handle-owned message.
// On a computation failure, earlier frames may already have been processed.
// Caller owns buffers and guarantees actual allocations match their sizes.
// No concurrent calls on one handle; no concurrent writes to frame buffers.
int rr_process_frames(void*,const std::uint8_t*,std::size_t,std::size_t,std::size_t,int,
                      double*,std::size_t,int*,std::size_t);
// Features are consecutive doubles: frame count * rr_feature_count(handle).
int rr_extract_features(void*,const std::uint8_t*,std::size_t,std::size_t,std::size_t,int,
                        double*,std::size_t);
}
