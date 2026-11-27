#pragma once
#include "../original/rings_cifar10/rings.hpp"
#include <algorithm>
#include <numeric>
namespace rings {
struct Graph {
 std::vector<std::uint64_t> ids;
 std::vector<std::size_t> row;
 std::vector<std::uint32_t> pre;
 std::vector<double> weight;
};
struct GraphConfig { double dt=.001, hz=10, gain=1, coupling=.4, width=Pi/2; };
class RingGraph {
 const Graph& g;
 GraphConfig c;
 std::vector<double> next, next_phase;
 std::vector<unsigned char> next_gate;
public:
 std::vector<double> activity, phase;
 std::vector<unsigned char> gate, silenced;
 explicit RingGraph(const Graph& graph, GraphConfig config={}):g(graph),c(config),
 next(g.ids.size()),next_phase(g.ids.size()),next_gate(g.ids.size()),
 activity(g.ids.size()),phase(g.ids.size()),gate(g.ids.size(),1),silenced(g.ids.size()) {
 if(g.ids.empty() || g.row.size()!=g.ids.size()+1 || g.row.front()!=0 ||
 g.row.back()!=g.pre.size() || g.pre.size()!=g.weight.size() ||
 !std::is_sorted(g.row.begin(),g.row.end())) throw std::runtime_error("Invalid CSR");
 for(auto p:g.pre) if(p>=g.ids.size()) throw std::runtime_error("Invalid index");
 for(auto w:g.weight) if(!std::isfinite(w)) throw std::runtime_error("Invalid weight");
 for(double v:{c.dt,c.hz,c.gain,c.coupling,c.width}) if(!std::isfinite(v)||v<0) throw std::runtime_error("Invalid config");
 if(c.dt==0||c.width>Pi||!std::isfinite(2*Pi*c.hz*c.dt)) throw std::runtime_error("Invalid time or width");
 }
 void silence(std::size_t i){silenced.at(i)=1;activity.at(i)=0;}
 void step(const std::vector<double>& external) {
 if(external.size()!=activity.size()) throw std::runtime_error("Invalid input size");
 int bad=0;
 const double increment=2*Pi*c.hz*c.dt;
 #pragma omp parallel for schedule(static) reduction(|:bad)
 for(std::int64_t i=0;i<static_cast<std::int64_t>(activity.size());++i){
 double sum=0;
 for(auto k=g.row[i];k<g.row[i+1];++k) sum+=g.weight[k]*activity[g.pre[k]];
 double u=external[i]+c.coupling*sum;
 double raw=phase[i]+increment+c.gain*u*c.dt;
 if(!std::isfinite(u)||!std::isfinite(raw)){bad=1;continue;}
 double a=wrap(raw);
 bool open=std::abs(a)<=c.width;
 next_phase[i]=a;next_gate[i]=open;
 next[i]=gated_value(std::max(0.0,u),open&&!silenced[i]);
 }
 if(bad) throw std::runtime_error("Nonfinite step");
 activity.swap(next);phase.swap(next_phase);gate.swap(next_gate);
 }
};
}
