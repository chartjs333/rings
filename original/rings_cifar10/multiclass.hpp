#pragma once
#include "rings.hpp"

namespace rings {
constexpr std::size_t Classes=10;
struct MulticlassReadout {
    std::size_t features=0;
    std::vector<double> weights;
    std::array<double,Classes> bias{};
    void validate()const{
        if(features==0||features>Steps*N||weights.size()!=Classes*features)
            throw std::runtime_error("Invalid multiclass shape");
        for(double w:weights)if(!std::isfinite(w))throw std::runtime_error("Invalid multiclass weight");
        for(double b:bias)if(!std::isfinite(b))throw std::runtime_error("Invalid multiclass bias");
    }
    static MulticlassReadout load(const std::string& path){
        std::ifstream in(path);in.imbue(std::locale::classic());
        if(!in)throw std::runtime_error("Cannot open multiclass readout");
        auto expect=[&](const std::string& wanted){std::string word;if(!(in>>word)||word!=wanted)throw std::runtime_error("Missing field: "+wanted);};
        expect("RINGS_CIFAR10_READOUT_V1");expect("classes");std::size_t k;
        if(!(in>>k)||k!=Classes)throw std::runtime_error("Expected ten classes");
        MulticlassReadout m;expect("features");
        if(!(in>>m.features)||!m.features||m.features>Steps*N)throw std::runtime_error("Invalid feature count");
        expect("weights");std::size_t n;
        if(!(in>>n)||n!=Classes*m.features)throw std::runtime_error("Invalid weight count");
        m.weights.resize(n);
        for(double& w:m.weights)if(!(in>>w))throw std::runtime_error("Invalid weight");
        expect("biases");if(!(in>>n)||n!=Classes)throw std::runtime_error("Invalid bias count");
        for(double& b:m.bias)if(!(in>>b))throw std::runtime_error("Invalid bias");
        std::string trailing;if(in>>trailing)throw std::runtime_error("Trailing readout data");
        m.validate();return m;
    }
    int evaluate(const double* input,double* scores)const{
        int best=0;
        for(std::size_t k=0;k<Classes;++k){
            double sum=0,correction=0;
            for(std::size_t j=0;j<features;++j){
                const double term=input[j]*weights[k*features+j],next=sum+term;
                correction+=std::abs(sum)>=std::abs(term)?(sum-next)+term:(term-next)+sum;sum=next;
            }
            scores[k]=(sum+correction)+bias[k];
            if(!std::isfinite(scores[k]))throw std::runtime_error("Non-finite class score");
            if(k&&scores[k]>scores[best])best=static_cast<int>(k);
        }
        return best;
    }
};
class MulticlassClassifier {
    Classifier extractor_;
    MulticlassReadout readout_;
public:
    MulticlassClassifier(Model features,MulticlassReadout readout)
        :extractor_(std::move(features)),readout_(std::move(readout)){
        readout_.validate();
        if(extractor_.feature_count()!=readout_.features)throw std::runtime_error("Extractor/readout dimension mismatch");
    }
    void reset(){extractor_.reset();}
    int score(const std::uint8_t* image,bool reset_before,double* scores){
        std::array<double,Steps*N> features;
        extractor_.run(image,ImageBytes,reset_before,features.data(),false);
        return readout_.evaluate(features.data(),scores);
    }
};
}
