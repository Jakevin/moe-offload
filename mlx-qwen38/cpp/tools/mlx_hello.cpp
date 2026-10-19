#include <mlx/mlx.h>
#include <iostream>
namespace mx = mlx::core;
int main() {
  auto a = mx::random::normal({4, 64}, mx::bfloat16);
  auto w = mx::random::normal({128, 64});
  auto q = mx::quantize(w, 32, 4);
  auto y = mx::quantized_matmul(a, q[0], q[1], q[2], true, 32, 4);
  auto f = mx::compile([](const std::vector<mx::array>& x) { return std::vector<mx::array>{mx::exp(x[0]) * 2}; });
  auto z = f({mx::ones({3})})[0];
  mx::eval(y, z);
  std::cout << "mlx " << mx::version() << " y=" << y.shape(0) << "x" << y.shape(1) << " z0=" << mx::sum(z).item<float>() << " gpu=" << mx::metal::is_available() << "\n";
}
