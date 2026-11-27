#include "multiclass.hpp"
#include <iostream>
#include <functional>
using namespace rings;
static void check(bool x){if(!x)throw std::runtime_error("Check failed");}
int main(){int passed=0;auto test=[&](const char* name,const std::function<void()>& fn){fn();++passed;std::cout<<"PASS "<<name<<'\n';};
    try{
        test("class-major coefficients and all ten scores",[]{MulticlassReadout m;m.features=2;m.weights.resize(20);for(std::size_t k=0;k<Classes;++k){m.weights[k*2]=double(k);m.weights[k*2+1]=-2;m.bias[k]=.5;}m.validate();double x[]={3,1},s[10];check(m.evaluate(x,s)==9);for(int k=0;k<10;++k)check(s[k]==3*k-1.5);});
        test("ties use first class",[]{MulticlassReadout m;m.features=1;m.weights.assign(10,0);double x[]={1},s[10];check(m.evaluate(x,s)==0);});
        test("bias-only and negative scores",[]{MulticlassReadout m;m.features=1;m.weights.assign(10,0);m.bias.fill(-2);m.bias[7]=-1;double x[]={0},s[10];check(m.evaluate(x,s)==7);});
        test("compensated cancellation",[]{MulticlassReadout m;m.features=3;m.weights.assign(30,0);m.weights[0]=1e16;m.weights[1]=1;m.weights[2]=-1e16;double x[]={1,1,1},s[10];check(m.evaluate(x,s)==0);check(s[0]==1);});
        test("invalid shape rejected",[]{MulticlassReadout m;m.features=2;m.weights.resize(19);bool threw=false;try{m.validate();}catch(const std::exception&){threw=true;}check(threw);});
        test("nonfinite readout rejected",[]{MulticlassReadout m;m.features=1;m.weights.assign(10,0);m.bias[3]=std::numeric_limits<double>::infinity();bool threw=false;try{m.validate();}catch(const std::exception&){threw=true;}check(threw);});
        test("extractor dimension checked",[]{Model m;m.scale.fill(1);m.readout_bins=2;m.weights.assign(2*N,0);MulticlassReadout r;r.features=1;r.weights.assign(10,0);bool threw=false;try{MulticlassClassifier c(m,r);}catch(const std::exception&){threw=true;}check(threw);});
        test("full native pipeline returns ten logits",[]{Model m;m.scale.fill(1);m.readout_bins=2;m.phase_gain=.25;m.weights.assign(2*N,0);MulticlassReadout r;r.features=2*N;r.weights.assign(Classes*2*N,0);r.bias[4]=2;MulticlassClassifier c(m,r);std::array<std::uint8_t,ImageBytes> image{};double s[10];check(c.score(image.data(),true,s)==4);for(int k=0;k<10;++k)check(s[k]==(k==4?2:0));});
        std::cout<<passed<<" multiclass tests passed\n";
    }catch(const std::exception& e){std::cerr<<"FAIL after "<<passed<<": "<<e.what()<<'\n';return 1;}
}
