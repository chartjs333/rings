// Reproducible float64 phase-kernel benchmark for the unmodified RingGraph model.
// Build: g++ -O3 -std=c++17 -fopenmp -ffp-contract=off cpp/benchmark_simd.cpp -o build/benchmark_simd
// No fast-math, approximated wrap, or reordered synaptic reductions are used.
#include "rings_graph.hpp"
#include <immintrin.h>
#include <omp.h>
#include <sched.h>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>

using namespace rings;

namespace {

// These functions vectorize independent neurons, retaining the scalar expression's
// operation order. wrap() and gated_value() remain the repository originals.
__attribute__((target("avx2"), noinline))
void phase4(const double* phase, const double* external, const double* sums,
            double increment, const GraphConfig& c, double* u, double* raw) {
    const auto v_u = _mm256_add_pd(_mm256_loadu_pd(external),
        _mm256_mul_pd(_mm256_set1_pd(c.coupling), _mm256_loadu_pd(sums)));
    const auto v_raw = _mm256_add_pd(
        _mm256_add_pd(_mm256_loadu_pd(phase), _mm256_set1_pd(increment)),
        _mm256_mul_pd(_mm256_mul_pd(_mm256_set1_pd(c.gain), v_u), _mm256_set1_pd(c.dt)));
    _mm256_storeu_pd(u, v_u);
    _mm256_storeu_pd(raw, v_raw);
}

__attribute__((target("avx512f"), noinline))
void phase8(const double* phase, const double* external, const double* sums,
            double increment, const GraphConfig& c, double* u, double* raw) {
    const auto v_u = _mm512_add_pd(_mm512_loadu_pd(external),
        _mm512_mul_pd(_mm512_set1_pd(c.coupling), _mm512_loadu_pd(sums)));
    const auto v_raw = _mm512_add_pd(
        _mm512_add_pd(_mm512_loadu_pd(phase), _mm512_set1_pd(increment)),
        _mm512_mul_pd(_mm512_mul_pd(_mm512_set1_pd(c.gain), v_u), _mm512_set1_pd(c.dt)));
    _mm512_storeu_pd(u, v_u);
    _mm512_storeu_pd(raw, v_raw);
}

template<int Lanes> class SimdGraph {
    const Graph& graph_;
    GraphConfig config_;
    std::vector<double> next_, next_phase_;
    std::vector<unsigned char> next_gate_;
public:
    std::vector<double> activity, phase;
    std::vector<unsigned char> gate, silenced;
    explicit SimdGraph(const Graph& graph, GraphConfig config={}) :
        graph_(graph), config_(config), next_(graph.ids.size()),
        next_phase_(graph.ids.size()), next_gate_(graph.ids.size()),
        activity(graph.ids.size()), phase(graph.ids.size()),
        gate(graph.ids.size(), 1), silenced(graph.ids.size()) {
        // Apply exactly the reference constructor's graph/config validation.
        RingGraph validate(graph, config);
    }
    void step(const std::vector<double>& external) {
        if (external.size()!=activity.size()) throw std::runtime_error("Invalid input size");
        const auto n=static_cast<std::int64_t>(activity.size());
        const auto blocks=(n+Lanes-1)/Lanes;
        const double increment=2*Pi*config_.hz*config_.dt;
        int bad=0;
        #pragma omp parallel for schedule(static) reduction(|:bad)
        for (std::int64_t block=0; block<blocks; ++block) {
            const auto first=block*Lanes;
            const auto count=std::min<std::int64_t>(Lanes,n-first);
            alignas(64) double sums[Lanes]={},u[Lanes]={},raw[Lanes]={};
            for (std::int64_t lane=0; lane<count; ++lane) {
                auto i=first+lane;
                double sum=0;
                for (auto k=graph_.row[i]; k<graph_.row[i+1]; ++k)
                    sum+=graph_.weight[k]*activity[graph_.pre[k]];
                sums[lane]=sum;
            }
            if (count==Lanes) {
                if constexpr (Lanes==4)
                    phase4(phase.data()+first,external.data()+first,sums,increment,config_,u,raw);
                else
                    phase8(phase.data()+first,external.data()+first,sums,increment,config_,u,raw);
            } else {
                for (std::int64_t lane=0; lane<count; ++lane) {
                    const auto i=first+lane;
                    u[lane]=external[i]+config_.coupling*sums[lane];
                    raw[lane]=phase[i]+increment+config_.gain*u[lane]*config_.dt;
                }
            }
            for (std::int64_t lane=0; lane<count; ++lane) {
                const auto i=first+lane;
                if (!std::isfinite(u[lane])||!std::isfinite(raw[lane])) {bad=1;continue;}
                const double a=wrap(raw[lane]);
                const bool open=std::abs(a)<=config_.width;
                next_phase_[i]=a;next_gate_[i]=open;
                next_[i]=gated_value(std::max(0.0,u[lane]),open&&!silenced[i]);
            }
        }
        if (bad) throw std::runtime_error("Nonfinite step");
        activity.swap(next_);phase.swap(next_phase_);gate.swap(next_gate_);
    }
};

void read_exact(std::istream& in, void* destination, std::size_t bytes) {
    if (bytes && !in.read(static_cast<char*>(destination),static_cast<std::streamsize>(bytes)))
        throw std::runtime_error("Truncated graph file");
}

Graph load_graph(const std::string& path) {
    static_assert(sizeof(std::size_t)==8 && sizeof(double)==8, "Requires 64-bit host");
    const std::uint16_t endian=1;
    if (*reinterpret_cast<const unsigned char*>(&endian)!=1)
        throw std::runtime_error("Graph format requires little-endian host");
    std::ifstream file(path,std::ios::binary);
    if(!file) throw std::runtime_error("Cannot open graph file");
    char magic[8];std::uint64_t n,m;
    read_exact(file,magic,8);
    if(std::memcmp(magic,"RINGSCSR",8)) throw std::runtime_error("Invalid graph magic");
    read_exact(file,&n,8);read_exact(file,&m,8);
    if (!n || n>=UINT32_MAX || m>(std::numeric_limits<std::size_t>::max()-24-n*16-8)/12)
        throw std::runtime_error("Invalid graph dimensions");
    file.seekg(0,std::ios::end);
    const auto size=file.tellg();
    const auto expected=24+n*8+(n+1)*8+m*4+m*8;
    if(size<0 || static_cast<std::uint64_t>(size)!=expected)
        throw std::runtime_error("Graph file size mismatch");
    file.seekg(24,std::ios::beg);
    Graph graph;graph.ids.resize(n);graph.row.resize(n+1);graph.pre.resize(m);graph.weight.resize(m);
    read_exact(file,graph.ids.data(),n*8);read_exact(file,graph.row.data(),(n+1)*8);
    read_exact(file,graph.pre.data(),m*4);read_exact(file,graph.weight.data(),m*8);
    RingGraph validate(graph);
    return graph;
}

template<class Net> void write_state(const Net& net,const std::string& path) {
    if(path.empty()) return;
    std::ofstream file(path,std::ios::binary);
    const std::uint64_t n=net.activity.size();
    file.write("RINGSSTA",8);file.write(reinterpret_cast<const char*>(&n),8);
    file.write(reinterpret_cast<const char*>(net.activity.data()),n*8);
    file.write(reinterpret_cast<const char*>(net.phase.data()),n*8);
    file.write(reinterpret_cast<const char*>(net.gate.data()),n);
    if(!file) throw std::runtime_error("Cannot write state file");
}

struct Comparison {
    double max_activity_error=0,max_phase_error=0;
    std::uint64_t activity_bit_mismatches=0,phase_bit_mismatches=0,gate_mismatches=0;
    template<class A,class B> void check(const A& a,const B& b) {
        for(std::size_t i=0;i<a.activity.size();++i) {
            if(!std::isfinite(a.activity[i])||!std::isfinite(b.activity[i])||
               !std::isfinite(a.phase[i])||!std::isfinite(b.phase[i]))
                throw std::runtime_error("Nonfinite validation state");
            max_activity_error=std::max(max_activity_error,std::abs(a.activity[i]-b.activity[i]));
            max_phase_error=std::max(max_phase_error,std::abs(a.phase[i]-b.phase[i]));
            activity_bit_mismatches+=std::memcmp(&a.activity[i],&b.activity[i],sizeof(double))!=0;
            phase_bit_mismatches+=std::memcmp(&a.phase[i],&b.phase[i],sizeof(double))!=0;
            gate_mismatches+=a.gate[i]!=b.gate[i];
        }
    }
    bool passed()const{return !activity_bit_mismatches&&!phase_bit_mismatches&&!gate_mismatches;}
    void json() const {
        std::cout<<"{\"max_activity_error\":"<<max_activity_error
                 <<",\"max_phase_error\":"<<max_phase_error
                 <<",\"activity_bit_mismatches\":"<<activity_bit_mismatches
                 <<",\"phase_bit_mismatches\":"<<phase_bit_mismatches
                 <<",\"gate_mismatches\":"<<gate_mismatches<<"}";
    }
};

void require_cpu(const std::string& mode) {
    __builtin_cpu_init();
    if((mode=="avx2" && !__builtin_cpu_supports("avx2")) ||
       (mode=="avx512" && !__builtin_cpu_supports("avx512f")))
        throw std::runtime_error("Requested SIMD is unavailable");
}

struct TeamInfo {
    int actual_threads=0;
    std::vector<int> cpu_ids;
    void json_fields() const {
        std::cout<<",\"actual_threads\":"<<actual_threads<<",\"thread_cpu_ids\":[";
        for(std::size_t i=0;i<cpu_ids.size();++i) {
            if(i)std::cout<<",";
            std::cout<<cpu_ids[i];
        }
        std::cout<<"]";
    }
};

// Observe the runtime team and placement outside the timed stepping region.
TeamInfo observe_team(int requested_threads) {
    TeamInfo team;
    #pragma omp parallel
    {
        #pragma omp single
        {
            team.actual_threads=omp_get_num_threads();
            team.cpu_ids.resize(team.actual_threads);
        }
        team.cpu_ids[omp_get_thread_num()]=sched_getcpu();
    }
    if(team.actual_threads!=requested_threads)
        throw std::runtime_error("Actual OpenMP team count differs from requested threads");
    if(std::find(team.cpu_ids.begin(),team.cpu_ids.end(),-1)!=team.cpu_ids.end())
        throw std::runtime_error("Cannot observe CPU placement");
    return team;
}

template<class Net> void benchmark(const Graph& graph,const std::vector<double>& input,
        const std::string& mode,int threads,std::uint64_t steps,std::uint64_t warmup,
        const std::string& output) {
    {
        Net warm(graph);
        for(std::uint64_t i=0;i<warmup;++i) warm.step(input);
    }
    Net net(graph);  // identical initial state, independent of warmup count
    const auto team=observe_team(threads);
    const auto begin=std::chrono::steady_clock::now();
    for(std::uint64_t i=0;i<steps;++i) net.step(input);
    const double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
    std::uint64_t active=0;bool finite=true;
    for(std::size_t i=0;i<net.activity.size();++i) {
        finite &= std::isfinite(net.activity[i])&&std::isfinite(net.phase[i]);
        active += net.activity[i]!=0;
    }
    write_state(net,output);
    std::cout<<"{\"benchmark_version\":1,\"mode\":\""<<mode<<"\",\"threads\":"<<threads;
    team.json_fields();
    std::cout<<",\"nodes\":"<<graph.ids.size()<<",\"edges\":"<<graph.pre.size()
        <<",\"steps\":"<<steps<<",\"warmup_steps\":"<<warmup
        <<",\"seconds\":"<<elapsed<<",\"steps_per_second\":"<<steps/elapsed
        <<",\"final_activity_sum\":"<<std::accumulate(net.activity.begin(),net.activity.end(),0.0)
        <<",\"final_phase_sum\":"<<std::accumulate(net.phase.begin(),net.phase.end(),0.0)
        <<",\"final_nonzero_activity\":"<<active<<",\"finite\":"<<(finite?"true":"false")<<"}\n";
}

bool validate(const Graph& graph,const std::vector<double>& input,std::uint64_t steps,int threads) {
    require_cpu("avx2");require_cpu("avx512");
    const auto team=observe_team(threads);
    RingGraph reference(graph);SimdGraph<4> avx2(graph);SimdGraph<8> avx512(graph);
    Comparison c2,c512;
    for(std::uint64_t i=0;i<steps;++i) {
        reference.step(input);avx2.step(input);avx512.step(input);
        c2.check(reference,avx2);c512.check(reference,avx512);
    }
    std::cout<<"{\"validation\":\"all_steps\",\"steps\":"<<steps<<",\"nodes\":"<<graph.ids.size()
             <<",\"threads\":"<<threads;
    team.json_fields();
    std::cout<<",\"avx2\":";c2.json();std::cout<<",\"avx512\":";c512.json();
    std::cout<<",\"passed\":"<<(c2.passed()&&c512.passed()?"true":"false")<<"}\n";
    return c2.passed()&&c512.passed();
}

bool self_test() {
    require_cpu("avx2");require_cpu("avx512");
    Graph graph;
    for(std::uint32_t i=0;i<19;++i) {
        graph.ids.push_back(1000+i);graph.row.push_back(graph.pre.size());
        for(std::uint32_t k=0;k<i%5;++k) {
            graph.pre.push_back((i*7+k*3)%19);
            graph.weight.push_back((k%2?-1.:1.)/(i%5));
        }
    }
    graph.row.push_back(graph.pre.size());
    const GraphConfig config{.003,13,.73,.61,.42};
    Comparison c2,c512;
    for(int threads:{1,4}) {
        omp_set_num_threads(threads);
        observe_team(threads);
        RingGraph reference(graph,config);SimdGraph<4> avx2(graph,config);SimdGraph<8> avx512(graph,config);
        const double phases[]={0,Pi,-Pi,std::nextafter(Pi,0.),std::nextafter(-Pi,0.),
            std::nextafter(Pi,4.),std::nextafter(-Pi,-4.),0x1p21,-0x1p21,1e100,-1e100};
        std::vector<double> input(19);
        for(std::size_t i=0;i<19;++i) {
            reference.phase[i]=avx2.phase[i]=avx512.phase[i]=phases[i%11];
            reference.activity[i]=avx2.activity[i]=avx512.activity[i]=(i%4)*.125;
            reference.silenced[i]=avx2.silenced[i]=avx512.silenced[i]=(i%7==0);
        }
        for(int step=0;step<97;++step) {
            for(std::size_t i=0;i<19;++i) input[i]=(static_cast<int>((i+step)%9)-4)*.2;
            reference.step(input);avx2.step(input);avx512.step(input);
            c2.check(reference,avx2);c512.check(reference,avx512);
        }
    }
    std::cout<<"{\"self_test\":\"tails_boundaries_huge_phases_signed_inputs_silencing_threads_1_4\",\"avx2\":";
    c2.json();std::cout<<",\"avx512\":";c512.json();
    std::cout<<",\"passed\":"<<(c2.passed()&&c512.passed()?"true":"false")<<"}\n";
    return c2.passed()&&c512.passed();
}

std::uint64_t parse_uint(const std::string& text) {
    if(text.empty()||text.find_first_not_of("0123456789")!=std::string::npos)
        throw std::runtime_error("Expected unsigned integer");
    return std::stoull(text);
}
}

int main(int argc,char**argv) {
    try {
        std::cout<<std::setprecision(17);
        std::string graph_path,mode="reference",output;
        std::uint64_t steps=400,warmup=50;int threads=1;
        bool do_validate=false,do_self_test=false;
        std::vector<std::uint64_t> stimuli;
        for(int i=1;i<argc;++i) {
            const std::string arg=argv[i];
            if(arg=="--validate") {do_validate=true;continue;}
            if(arg=="--self-test") {do_self_test=true;continue;}
            if(i+1>=argc)throw std::runtime_error("Missing argument value");
            const std::string value=argv[++i];
            if(arg=="--graph")graph_path=value;
            else if(arg=="--mode")mode=value;
            else if(arg=="--output")output=value;
            else if(arg=="--steps")steps=parse_uint(value);
            else if(arg=="--warmup")warmup=parse_uint(value);
            else if(arg=="--threads") {
                const auto t=parse_uint(value);
                if(!t||t>1024)throw std::runtime_error("Invalid thread count");
                threads=static_cast<int>(t);
            }
            else if(arg=="--stimulus")stimuli.push_back(parse_uint(value));
            else throw std::runtime_error("Unknown argument: "+arg);
        }
        omp_set_dynamic(0);omp_set_num_threads(threads);
        if(do_self_test) return self_test()?0:1;
        if(graph_path.empty()||stimuli.empty()||!steps)
            throw std::runtime_error("Required: --graph PATH --stimulus ID [--stimulus ID ...] [--mode reference|avx2|avx512] [--threads N] [--steps N] [--warmup N] [--output PATH] [--validate]; or --self-test");
        if(mode!="reference"&&mode!="avx2"&&mode!="avx512")throw std::runtime_error("Unknown mode");
        require_cpu(mode);
        const Graph graph=load_graph(graph_path);
        std::vector<double> input(graph.ids.size());
        for(auto id:stimuli) {
            const auto it=std::find(graph.ids.begin(),graph.ids.end(),id);
            if(it==graph.ids.end())throw std::runtime_error("Unknown stimulus ID");
            input[it-graph.ids.begin()]=1.;
        }
        if(do_validate)return validate(graph,input,steps,threads)?0:1;
        if(mode=="reference")benchmark<RingGraph>(graph,input,mode,threads,steps,warmup,output);
        else if(mode=="avx2")benchmark<SimdGraph<4>>(graph,input,mode,threads,steps,warmup,output);
        else benchmark<SimdGraph<8>>(graph,input,mode,threads,steps,warmup,output);
        return 0;
    } catch(const std::exception& error) {std::cerr<<error.what()<<"\n";return 1;}
}
