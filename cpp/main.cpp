#include "rings_graph.hpp"
#include <unordered_map>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <charconv>
using namespace rings;
static std::uint64_t id(const std::string& s){
 std::uint64_t v=0;auto r=std::from_chars(s.data(),s.data()+s.size(),v);
 if(r.ec!=std::errc{}||r.ptr!=s.data()+s.size())throw std::runtime_error("Invalid ID");return v;
}
static std::vector<std::string> split(std::string s){
 if(!s.empty()&&s.back()=='\r')s.pop_back();
 std::vector<std::string> v;std::size_t p=0;
 for(;;){auto q=s.find(',',p);v.push_back(s.substr(p,q-p));if(q==std::string::npos)return v;p=q+1;}
}
static Graph load(const char* nodes,const char* edges){
 Graph g;std::unordered_map<std::uint64_t,std::uint32_t> idx;std::vector<int> signs;
 std::ifstream n(nodes),e(edges);std::string line;
 if(!std::getline(n,line)||split(line)!=std::vector<std::string>{"neuron_id","sign"})throw std::runtime_error("Invalid nodes header");
 while(std::getline(n,line)){auto v=split(line);if(v.size()!=2)throw std::runtime_error("Invalid node");
 auto x=id(v[0]);if(g.ids.size()>=UINT32_MAX||!idx.emplace(x,g.ids.size()).second)throw std::runtime_error("Duplicate or excess nodes");
 if(v[1]!="1"&&v[1]!="-1"&&v[1]!="0")throw std::runtime_error("Invalid sign");
 g.ids.push_back(x);signs.push_back(std::stoi(v[1]));}
 struct Edge{std::uint32_t post,pre;double w;};std::vector<Edge> es;
 if(!std::getline(e,line)||split(line)!=std::vector<std::string>{"pre_id","post_id","syn_count"})throw std::runtime_error("Invalid edges header");
 while(std::getline(e,line)){auto v=split(line);if(v.size()!=3)throw std::runtime_error("Invalid edge");
 auto p=idx.at(id(v[0])),q=idx.at(id(v[1]));std::size_t used=0;double w=std::stod(v[2],&used);
 if(used!=v[2].size()||!std::isfinite(w)||w<0)throw std::runtime_error("Invalid count");
 es.push_back({q,p,w*signs[p]});}
 if(n.bad()||e.bad())throw std::runtime_error("Read failure");
 std::sort(es.begin(),es.end(),[](auto a,auto b){return a.post<b.post||(a.post==b.post&&a.pre<b.pre);});
 g.row.resize(g.ids.size()+1);std::size_t k=0;
 for(std::size_t i=0;i<g.ids.size();++i){
 double norm=0;auto start=g.weight.size();
 while(k<es.size()&&es[k].post==i){auto p=es[k].pre;double w=0;
 do{w+=es[k++].w;}while(k<es.size()&&es[k].post==i&&es[k].pre==p);
 if(w!=0){g.pre.push_back(p);g.weight.push_back(w);norm+=std::abs(w);}}
 if(!std::isfinite(norm))throw std::runtime_error("Overflow");
 for(auto j=start;j<g.weight.size();++j)g.weight[j]/=std::max(1.0,norm);
 g.row[i+1]=g.weight.size();}
 return g;
}
int main(int argc,char**argv){try{
 if(argc<5){std::cerr<<"Usage: rings_fly_cpp nodes.csv connections.csv steps stimulus_id [stimulus_id ...]\n";return 2;}
 auto g=load(argv[1],argv[2]);auto steps=id(argv[3]);if(!steps)throw std::runtime_error("Zero steps");
 RingGraph net(g);std::vector<double> input(g.ids.size());
 for(int j=4;j<argc;++j){auto it=std::find(g.ids.begin(),g.ids.end(),id(argv[j]));if(it==g.ids.end())throw std::runtime_error("Unknown stimulus");input[it-g.ids.begin()]=1;}
 auto t=std::chrono::steady_clock::now();
 for(std::uint64_t i=0;i<steps;++i)net.step(input);
 double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();
 std::cout<<std::setprecision(17)<<"nodes="<<g.ids.size()<<" edges="<<g.pre.size()<<" steps="<<steps<<" seconds="<<seconds<<" steps_per_second="<<steps/seconds<<" final_activity_sum="<<std::accumulate(net.activity.begin(),net.activity.end(),0.0)<<"\n";
 return 0;}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
