# LinearResponseVariationalBayes.cpp

This contains a C++ library with tools for doing linear response variational
Bayes calculations, especially using the Stan autodiff libraries.
It will require Stan and Stan math (as well as Eigen and boost, which are
by default installed with Stan math).

* [Stan repo](https://github.com/stan-dev/stan)
* [Stan math repo](https://github.com/stan-dev/math)

Install these libraries to wherever you keep your git repos.  Cmake will work
automatically if you set an environment variable to your git repo location,
e.g. in bash:

```
export GIT_REPO_LOC=/full/path/to/your/git/repos
```

Finally, in a build directory, run cmake then make install:

```bash
cd $GIT_REPO_LOC/LinearResponseVariationalBayes.cpp
mkdir build
cd build
cmake ..
sudo make install
```

If there are problems, check that CMakeLists.txt in the root directory has the
correct directory locations for the libraries.
