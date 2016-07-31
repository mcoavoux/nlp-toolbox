
#include "arrayfire.h"

using namespace std;

//http://colah.github.io/posts/2014-07-Conv-Nets-Modular/
//http://colah.github.io/posts/2014-07-Understanding-Convolutions/

af::array myconvolution(af::array &x, af::array &f){

  unsigned N = x.dims(0);
  af::array c = af::constant(0,N);

  for(int i = 0; i < N ; ++i){
    c[i] = 
  }
}


int main(){
  af::array xinput = af::iota(af::dim4(3));
  af::array xfilter = af::constant(0,3);
  xfilter(1) = 2;
  af::array h = af::convolve(xinput,xfilter,AF_CONV_EXPAND); // AF_CONV_DEFAULT
  af_print(h);


  return 0;
}
