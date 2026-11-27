#pragma once
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <locale>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace rings {
constexpr std::size_t H=16,W=16,C=3,N=H*W*C,Steps=4,ImageBytes=32*32*3;
constexpr double Pi=3.141592653589793238462643383279502884;
using Field=std::array<double,N>;
constexpr std::size_t index(std::size_t r,std::size_t c,std::size_t k) { return (r*W+c)*C+k; }
inline double wrap(double a) {
    constexpr double TwoPi=2.0*Pi,InvTwoPi=1.0/TwoPi;
    if(!std::isfinite(a))throw std::runtime_error("Non-finite angle");
    // Fast reduction for operational angles. Large finite doubles need robust
    // reduction: subtracting a rounded huge multiple can lose every low bit.
    double result=std::abs(a)<=0x1p20
        ? a-TwoPi*std::floor((a+Pi)*InvTwoPi)
        : std::remainder(a,TwoPi);
    // Rounding at nextafter(+/-Pi) can escape the half-open interval.
    if(result < -Pi)result+=TwoPi;
    if(result >= Pi)result-=TwoPi;
    return result;
}

inline double gated_value(double value,bool open) {
    // Alias-safe bit select: no data-dependent gate branch, closed output +0.
    static_assert(sizeof(double)==sizeof(std::uint64_t)&&std::numeric_limits<double>::is_iec559,
                  "Requires IEEE-754 binary64");
    std::uint64_t bits;std::memcpy(&bits,&value,sizeof(bits));
    bits&=std::uint64_t(0)-static_cast<std::uint64_t>(open);
    double result;std::memcpy(&result,&bits,sizeof(result));return result;
}

struct Model {
    Field mean{},scale{},angles{};
    std::vector<double> weights;
    double bias=0,half_width=Pi/2,edge_weight=.4,color_weight=0,velocity=Pi/2,phase_gain=0;
    // 0: legacy fixed-mask packing; 2: pairwise temporal means; 4: all steps.
    std::size_t readout_bins=0;
    void validate() const {
        if(!std::isfinite(half_width)||half_width<0||half_width>Pi||!std::isfinite(velocity)||
           !std::isfinite(edge_weight)||edge_weight<0||edge_weight>1||
           !std::isfinite(color_weight)||color_weight<0||color_weight>2||!std::isfinite(bias))
            throw std::runtime_error("Invalid filter configuration");
        if(!std::isfinite(phase_gain)||phase_gain<0||phase_gain>4||
           (readout_bins!=0&&readout_bins!=2&&readout_bins!=4)||
           (readout_bins==0&&phase_gain!=0)||
           (readout_bins!=0&&weights.size()!=readout_bins*N))
            throw std::runtime_error("Invalid phase/readout configuration");
        for(std::size_t i=0;i<N;++i)
            if(!std::isfinite(mean[i])||!std::isfinite(scale[i])||scale[i]<=0||!std::isfinite(angles[i]))
                throw std::runtime_error("Invalid normalizer/phase");
        if(weights.empty()||weights.size()>Steps*N)throw std::runtime_error("Invalid readout size");
        for(double w:weights)if(!std::isfinite(w))throw std::runtime_error("Invalid readout weight");
    }
    static Model load(const std::string& path) {
        std::ifstream in(path);in.imbue(std::locale::classic());
        if(!in)throw std::runtime_error("Cannot open model: "+path);
        auto expect=[&](const std::string& wanted){std::string word;if(!(in>>word)||word!=wanted)throw std::runtime_error("Missing field: "+wanted);};
        auto number=[&](){double x;if(!(in>>x)||!std::isfinite(x))throw std::runtime_error("Invalid number");return x;};
        std::string version;in>>version;
        if(version!="RINGS2D_CPP_V1"&&version!="RINGS2D_CPP_V2"&&version!="RINGS2D_CPP_V3")throw std::runtime_error("Unsupported model format");
        Model m;expect("half_width");m.half_width=number();
        if(version=="RINGS2D_CPP_V1") {expect("coupling");m.edge_weight=number();m.color_weight=0;}
        else {expect("edge_weight");m.edge_weight=number();expect("color_weight");m.color_weight=number();}
        expect("velocity");m.velocity=number();
        if(version=="RINGS2D_CPP_V3"){
            expect("phase_gain");m.phase_gain=number();expect("readout_bins");
            if(!(in>>m.readout_bins)||(m.readout_bins!=2&&m.readout_bins!=4))
                throw std::runtime_error("Expected 2 or 4 readout bins");
        }
        for(const std::string name:{"mean","scale","angles"}) {
            expect(name);std::size_t n;if(!(in>>n)||n!=N)throw std::runtime_error("Invalid field length");
            Field& field=name=="mean"?m.mean:name=="scale"?m.scale:m.angles;
            for(double& x:field)x=number();
        }
        expect("weights");std::size_t n;if(!(in>>n)||!n||n>Steps*N)throw std::runtime_error("Invalid readout length");
        m.weights.resize(n);for(double& x:m.weights)x=number();expect("bias");m.bias=number();
        std::string trailing;if(in>>trailing)throw std::runtime_error("Trailing model data");
        m.validate();return m;
    }
};

struct Input {Field signal{},color{};};
struct Step {Field output{};std::array<std::uint16_t,N> active{};std::size_t count=0;};

class RingGrid {
    Field initial_{},state_{},phase_{};
    double width_,edge_,color_,velocity_,phase_gain_;
public:
    RingGrid(const Field& angles,double width=Pi/2,double edge=.4,double color=0,double velocity=Pi/2,double phase_gain=0)
        :initial_(angles),width_(width),edge_(edge),color_(color),velocity_(velocity),phase_gain_(phase_gain){reset();}
    void reset(){state_.fill(0);for(std::size_t i=0;i<N;++i)phase_[i]=wrap(initial_[i]);}
    const Field& state()const{return state_;}
    static std::array<double,3> color_deviation(const std::array<double,3>& rgb) {
        const double mean=(rgb[0]+rgb[1]+rgb[2])/3.0;
        return {rgb[0]-mean,rgb[1]-mean,rgb[2]-mean};
    }
    static Field contour(const Field& field) {
        Field out{};
        for(std::size_t r=0;r<H;++r){
            const auto up=r?r-1:1,down=r+1<H?r+1:H-2;
            for(std::size_t c=0;c<W;++c){
                const auto left=c?c-1:1,right=c+1<W?c+1:W-2;
                for(std::size_t k=0;k<C;++k){
                    const auto i=index(r,c,k);
                    const double surround=(field[index(up,c,k)]+field[index(down,c,k)]
                                           +field[index(r,left,k)]+field[index(r,right,k)])*.25;
                    out[i]=field[i]-surround;
                }
            }
        }
        return out;
    }
    // Dense readouts instantiate CollectActive=false, eliminating mask packing.
    template<bool CollectActive=true>
    Step step(const Input& input) {
        Step result;Field next_phase{};
        for(std::size_t r=0;r<H;++r){
            const auto up=r?r-1:1,down=r+1<H?r+1:H-2;
            for(std::size_t c=0;c<W;++c){
                const auto left=c?c-1:1,right=c+1<W?c+1:W-2;
                for(std::size_t k=0;k<C;++k){
                    const auto i=index(r,c,k);
                    double angle=0;
                    if(phase_gain_==0){
                        angle=wrap(phase_[i]+velocity_);
                        next_phase[i]=angle;
                        if(std::abs(angle)>width_)continue;
                    }
                    // Flat direct indices, four same-channel spatial neighbours, OLD state.
                    const double surround=(state_[index(up,c,k)]+state_[index(down,c,k)]
                                           +state_[index(r,left,k)]+state_[index(r,right,k)])*.25;
                    const double edge=state_[i]-surround;
                    double value=input.signal[i]+edge_*edge;
                    if(color_!=0)value+=color_*input.color[i];
                    if(!std::isfinite(value))throw std::runtime_error("Non-finite hybrid signal");
                    if(phase_gain_!=0){
                        // The candidate signal is evaluated even for a closed gate.
                        const double phase=phase_[i]+velocity_+phase_gain_*value;
                        if(!std::isfinite(phase))throw std::runtime_error("Non-finite phase");
                        angle=wrap(phase);next_phase[i]=angle;
                    }
                    const bool open=std::abs(angle)<=width_;
                    result.output[i]=gated_value(value,open);
                    if constexpr(CollectActive){
                        if(open)result.active[result.count++]=static_cast<std::uint16_t>(i);
                    }
                }
            }
        }
        state_=result.output;phase_=next_phase;return result;
    }
};

class Classifier {
    Model model_;
    RingGrid rings_;
    std::array<std::uint16_t,Steps*N> expected_{};
    std::array<std::size_t,Steps> counts_{};
public:
    explicit Classifier(Model model):model_(std::move(model)),
        rings_(model_.angles,model_.half_width,model_.edge_weight,model_.color_weight,model_.velocity,model_.phase_gain){
        model_.validate();Input zero{};std::size_t offset=0;
        if(model_.readout_bins!=0)return;
        for(std::size_t t=0;t<Steps;++t){
            const Step step=rings_.step(zero);counts_[t]=step.count;
            for(std::size_t j=0;j<step.count;++j)expected_[offset++]=step.active[j];
        }
        if(offset!=model_.weights.size())throw std::runtime_error("Readout does not match gates");
        reset();
    }
    std::size_t feature_count()const{return model_.weights.size();}
    void reset(){rings_.reset();}
    Input preprocess(const std::uint8_t* image,std::size_t size)const{
        if(!image||size!=ImageBytes)throw std::runtime_error("Expected 3072 RGB bytes");
        Input input;
        for(std::size_t r=0;r<H;++r)for(std::size_t c=0;c<W;++c){
            std::array<double,3> rgb{};
            for(std::size_t k=0;k<C;++k){
                const auto p=((2*r)*32+2*c)*3+k;
                rgb[k]=(static_cast<double>(image[p])+image[p+96]+image[p+3]+image[p+99])/(4*255);
            }
            // Brightness is the arithmetic RGB mean BEFORE per-channel standardization.
            const auto deviation=RingGrid::color_deviation(rgb);
            for(std::size_t k=0;k<C;++k){
                const auto i=index(r,c,k);
                input.signal[i]=(rgb[k]-model_.mean[i])/model_.scale[i];
                input.color[i]=deviation[k]/model_.scale[i];
            }
        }
        return input;
    }
    double run(const std::uint8_t* image,std::size_t size,bool reset_before,double* features=nullptr,bool readout=true){
        const Input input=preprocess(image,size);if(reset_before)reset();
        if(model_.readout_bins!=0){
            // Stable global node indices; inactive outputs remain exactly zero.
            // bins=4 -> weights[t*N+i]; bins=2 -> weights[(t/2)*N+i].
            std::array<double,Steps*N> pooled{};
            const auto group=Steps/model_.readout_bins;
            const double factor=1.0/static_cast<double>(group);
            for(std::size_t t=0;t<Steps;++t){
                const Step step=rings_.step<false>(input);
                for(std::size_t i=0;i<N;++i){
                    pooled[(t/group)*N+i]+=step.output[i]*factor;
                }
            }
            double sum=0,correction=0;
            for(std::size_t f=0;f<model_.weights.size();++f){
                if(features)features[f]=pooled[f];
                if(readout){
                    const double term=pooled[f]*model_.weights[f],next=sum+term;
                    correction+=std::abs(sum)>=std::abs(term)?(sum-next)+term:(term-next)+sum;sum=next;
                }
            }
            return (sum+correction)+model_.bias;
        }
        double sum=0,correction=0;std::size_t f=0;
        for(std::size_t t=0;t<Steps;++t){
            const Step step=rings_.step(input);
            if(step.count!=counts_[t])throw std::runtime_error("Gate schedule drifted; reset memory");
            for(std::size_t j=0;j<step.count;++j){
                const auto i=step.active[j];if(i!=expected_[f])throw std::runtime_error("Gate order drifted; reset memory");
                const double value=step.output[i];if(features)features[f]=value;
                if(readout){
                    const double term=value*model_.weights[f],next=sum+term;
                    correction+=std::abs(sum)>=std::abs(term)?(sum-next)+term:(term-next)+sum;sum=next;
                }
                ++f;
            }
        }
        return (sum+correction)+model_.bias;
    }
    double score(const std::uint8_t* image,std::size_t size,bool reset_before=true){return run(image,size,reset_before);}
    void score_batch(const std::uint8_t* images,std::size_t count,std::size_t stride,bool reset_each,double* scores,int* labels){
        for(std::size_t n=0;n<count;++n){
            const double value=score(images+n*stride,ImageBytes,reset_each);
            scores[n]=value;labels[n]=value>0?1:0;
        }
    }
    void features_batch(const std::uint8_t* images,std::size_t count,std::size_t stride,bool reset_each,double* features){
        const auto size=feature_count();
        for(std::size_t n=0;n<count;++n)run(images+n*stride,ImageBytes,reset_each,features+n*size,false);
    }
};
}
