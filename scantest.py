import numpy as np
import theano
import theano.tensor as T

def power_of_2(previous_powers, coefficients):
    new_values = previous_powers*coefficients
    index = T.argmax(new_values)
    return new_values, theano.scan_module.until(T.eq(index, T.constant(0)))

coefficients = T.vector("coefficients")
init_values = T.vector("init_values")

values, _ = theano.scan(power_of_2,
                        outputs_info = [init_values],
                        non_sequences = coefficients,
                        n_steps = 1024)

f = theano.function([init_values, coefficients], values)

print(f(3.0 * np.arange(1,4).astype("float32"), np.asarray([2, 1.8, 1.6]).astype("float32")))
