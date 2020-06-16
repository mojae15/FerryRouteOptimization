#include <fdeep/fdeep.hpp>

int main(){

    const auto model = fdeep::load_model("fdeep_model.json");
    const auto result = model.predict(
        {fdeep::tensor(fdeep::tensor_shape(static_cast<std::size_t>(7)), 
        {1, 2, 3, 4, 5, 6, 7})});
    
    std::cout << fdeep::show_tensors(result) << std::endl;


}