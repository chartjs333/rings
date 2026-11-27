#include "multiclass.hpp"
struct MCHandle {
    rings::MulticlassClassifier classifier;
    std::string error;
    MCHandle(const char* a,const char* b):classifier(rings::Model::load(a),rings::MulticlassReadout::load(b)){}
};
extern "C" {
void* mc_create(const char* features,const char* readout,char* error,std::size_t capacity){
    try{if(!features||!readout)throw std::runtime_error("Null model path");return new MCHandle(features,readout);}
    catch(const std::exception& e){if(error&&capacity){std::strncpy(error,e.what(),capacity-1);error[capacity-1]=0;}return nullptr;}
}
void mc_destroy(void* p){delete static_cast<MCHandle*>(p);}
const char* mc_error(void* p){return p?static_cast<MCHandle*>(p)->error.c_str():"Null multiclass handle";}
int mc_reset(void* p){if(!p)return -1;auto* h=static_cast<MCHandle*>(p);h->classifier.reset();h->error.clear();return 0;}
int mc_process(void* p,const std::uint8_t* data,std::size_t bytes,std::size_t count,std::size_t stride,int reset,
               double* scores,std::size_t score_capacity,int* labels,std::size_t label_capacity){
    if(!p)return -1;
    auto* h=static_cast<MCHandle*>(p);
    try{
        const auto limit=std::numeric_limits<std::size_t>::max();
        if(reset!=0&&reset!=1)throw std::runtime_error("Invalid memory mode");
        if(count){
            if(!data||stride<rings::ImageBytes||count-1>(limit-rings::ImageBytes)/stride||
               bytes<(count-1)*stride+rings::ImageBytes)throw std::runtime_error("Invalid frame buffer");
            if(count>limit/rings::Classes||!scores||!labels||score_capacity<count*rings::Classes||label_capacity<count)
                throw std::runtime_error("Invalid output buffer");
        }
        for(std::size_t n=0;n<count;++n)labels[n]=h->classifier.score(data+n*stride,reset!=0,scores+n*rings::Classes);
        h->error.clear();return 0;
    }catch(const std::exception& e){h->error=e.what();return -1;}
}
}
