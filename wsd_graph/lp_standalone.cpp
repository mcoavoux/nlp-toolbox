#include "lp.h"
#include <fstream>

int main(){
    WSDdataset data("/Users/bcrabbe/Desktop/LP.in");
    WSDGraph G;
    Proximity *prox = new Proximity();
    data.get_graph(G,prox);
    delete prox;
    //G.eps_nn(0.01);
    G.knn(3);
    G.label_propagation(0.0000001);
    wofstream out("json.txt");
    G.json(out);
    out.close();
}
