#include "module1/module1c1.hpp"
#include "module1/module1c2.hpp"
#include "module2/module2c1.hpp"
#include "module2/module2c2.hpp"

int main(){
    mod1c1 m1c1; m1c1.foo();
    mod1c1 m1c2; m1c2.foo();
    mod2c1 m2c1; m2c1.foo();
    mod2c2 m2c2; m2c2.foo();
}