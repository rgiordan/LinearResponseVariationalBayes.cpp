# LinearResponseVariationalBayes.cpp

This contains a C++ library with tools for doing linear response variational
Bayes calculations, especially using the Stan autodiff libraries.

First, it requires an environment variable called GIT_REPO_LOC to be
set to the location of git repositories.  For example, in your .bashrc,
put the line

export GIT_REPO_LOC="$HOME/git_repos/"

It will require stan and stan/math (as well as Eigen and boost, which are
by default installed with stan/math).

https://github.com/stan-dev/stan
https://github.com/stan-dev/math

Install these libraries to the git repo location.  Finally, change to the
src directory in this repository and run cmake then make install:

cd $GIT_REPO_LOC/LinearResponseVariationalBayes.cpp/src
cmake .
sudo make install
