# Test Protocol (work in progress)

For the time being, we'll use the [`pytest`](https://docs.pytest.org/en/latest/) unit testing framework in Python. This is a simple alternative to the Python standard library's `unittest` module. 

## Organization

The layout of the `test` directory and sub-directories should closely correspond to that of the `spiketorch` library; i.e., there should be separate files containing unit tests for files found in `spiketorch`. On the other hand, integration, performance, and other types of tests may have separate folders at the top level of the `test` directory.

## Unit tests

Every function should be tested as a unit, to the extent that it is possible, and as many corner cases which can be feasibly covered should be.

## Integration tests

Integration tests (groups of components are combined to produce an output) will also be very important. Simple networks should be built, and pre-computed outputs of short simulations should be compared to simulation results. Not all corner cases are easily covered here, but those which are in use in experimental scripts should be tested thoroughly in this way.
