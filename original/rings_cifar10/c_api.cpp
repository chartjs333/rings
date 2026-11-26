#include "rings.hpp"
#include "c_api.h"
#include <cstring>

struct Handle { rings::Classifier classifier;std::string error;explicit Handle(const char* path):classifier(rings::Model::load(path)){} };
static void validate_frames(const std::uint8_t* data,std::size_t bytes,std::size_t count,std::size_t stride,int reset){
    if(reset!=0&&reset!=1)throw std::runtime_error("reset_each must be 0 or 1");
    if(!count)return;
    if(!data||stride<rings::ImageBytes||count-1>(std::numeric_limits<std::size_t>::max()-rings::ImageBytes)/stride)
        throw std::runtime_error("Invalid frame buffer or stride");
    if(bytes<(count-1)*stride+rings::ImageBytes)throw std::runtime_error("Truncated frame buffer");
}
extern "C" {
void* rr_create(const char* path,char* error,std::size_t capacity){
    try{if(!path)throw std::runtime_error("Null model path");return new Handle(path);}
    catch(const std::exception& e){if(error&&capacity){std::strncpy(error,e.what(),capacity-1);error[capacity-1]=0;}return nullptr;}
}
void rr_destroy(void* p){delete static_cast<Handle*>(p);}
const char* rr_error(void* p){return p?static_cast<Handle*>(p)->error.c_str():"Null classifier";}
int rr_reset(void* p){if(!p)return -1;auto* h=static_cast<Handle*>(p);h->classifier.reset();h->error.clear();return 0;}
std::size_t rr_feature_count(void* p){return p?static_cast<Handle*>(p)->classifier.feature_count():0;}
int rr_process_frames(void* p,const std::uint8_t* data,std::size_t bytes,std::size_t count,std::size_t stride,
                      int reset,double* scores,std::size_t score_capacity,int* labels,std::size_t label_capacity){
    if(!p)return -1;
    auto* h=static_cast<Handle*>(p);
    try{
        validate_frames(data,bytes,count,stride,reset);
        if(count&&(!scores||!labels||score_capacity<count||label_capacity<count))throw std::runtime_error("Output buffer too small");
        h->classifier.score_batch(data,count,stride,reset!=0,scores,labels);h->error.clear();return 0;
    }catch(const std::exception& e){h->error=e.what();return -1;}
}
int rr_predict(void* p,const std::uint8_t* data,std::size_t bytes,int reset,double* score,int* label){
    if(p&&bytes!=rings::ImageBytes){static_cast<Handle*>(p)->error="Expected exactly one RGB frame";return -1;}
    return rr_process_frames(p,data,bytes,1,rings::ImageBytes,reset,score,1,label,1);
}
int rr_predict_batch(void* p,const std::uint8_t* data,std::size_t bytes,std::size_t count,int reset,double* scores,int* labels){
    return rr_process_frames(p,data,bytes,count,rings::ImageBytes,reset,scores,count,labels,count);
}
int rr_extract_features(void* p,const std::uint8_t* data,std::size_t bytes,std::size_t count,std::size_t stride,
                        int reset,double* features,std::size_t capacity){
    if(!p)return -1;
    auto* h=static_cast<Handle*>(p);
    try{
        validate_frames(data,bytes,count,stride,reset);const auto size=h->classifier.feature_count();
        if(count>(std::numeric_limits<std::size_t>::max()/size)||capacity<count*size||(count&&!features))
            throw std::runtime_error("Feature buffer too small");
        h->classifier.features_batch(data,count,stride,reset!=0,features);h->error.clear();return 0;
    }catch(const std::exception& e){h->error=e.what();return -1;}
}
}
