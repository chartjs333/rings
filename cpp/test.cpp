#include "rings_graph.hpp"
#include <iostream>
int main(){
 rings::Graph g{{1,2,3},{0,0,1,3},{0,0,1},{1,-.5,.5}};
 rings::GraphConfig c;c.hz=0;c.gain=0;c.coupling=1;
 rings::RingGraph n(g,c);
 n.step({2,0,0});if(n.activity!=std::vector<double>{2,0,0})return 1;
 n.step({0,0,0});if(n.activity!=std::vector<double>{0,2,0})return 2;
 n.step({0,0,0});if(n.activity!=std::vector<double>{0,0,1})return 3;
 n.silence(0);n.step({2,0,0});if(n.activity[0]!=0)return 4;
 c.gain=1;c.dt=.1;c.width=.1;rings::RingGraph p(g,c);p.step({2,0,0});
 if(p.gate[0]||p.activity[0]!=0||std::abs(p.phase[0]-.2)>1e-12)return 5;
 std::cout<<"PASS\n";
}
