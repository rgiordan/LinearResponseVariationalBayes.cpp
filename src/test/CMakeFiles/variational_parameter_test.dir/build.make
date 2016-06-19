# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.2

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/rgiordan/Documents/git_repos/LinearResponseVariationalBayes.cpp/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/rgiordan/Documents/git_repos/LinearResponseVariationalBayes.cpp/src

# Include any dependencies generated for this target.
include test/CMakeFiles/variational_parameter_test.dir/depend.make

# Include the progress variables for this target.
include test/CMakeFiles/variational_parameter_test.dir/progress.make

# Include the compile flags for this target's objects.
include test/CMakeFiles/variational_parameter_test.dir/flags.make

test/CMakeFiles/variational_parameter_test.dir/variational_parameter_test.cpp.o: test/CMakeFiles/variational_parameter_test.dir/flags.make
test/CMakeFiles/variational_parameter_test.dir/variational_parameter_test.cpp.o: test/variational_parameter_test.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/rgiordan/Documents/git_repos/LinearResponseVariationalBayes.cpp/src/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object test/CMakeFiles/variational_parameter_test.dir/variational_parameter_test.cpp.o"
	cd /home/rgiordan/Documents/git_repos/LinearResponseVariationalBayes.cpp/src/test && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/variational_parameter_test.dir/variational_parameter_test.cpp.o -c /home/rgiordan/Documents/git_repos/LinearResponseVariationalBayes.cpp/src/test/variational_parameter_test.cpp

test/CMakeFiles/variational_parameter_test.dir/variational_parameter_test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/variational_parameter_test.dir/variational_parameter_test.cpp.i"
	cd /home/rgiordan/Documents/git_repos/LinearResponseVariationalBayes.cpp/src/test && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/rgiordan/Documents/git_repos/LinearResponseVariationalBayes.cpp/src/test/variational_parameter_test.cpp > CMakeFiles/variational_parameter_test.dir/variational_parameter_test.cpp.i

test/CMakeFiles/variational_parameter_test.dir/variational_parameter_test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/variational_parameter_test.dir/variational_parameter_test.cpp.s"
	cd /home/rgiordan/Documents/git_repos/LinearResponseVariationalBayes.cpp/src/test && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/rgiordan/Documents/git_repos/LinearResponseVariationalBayes.cpp/src/test/variational_parameter_test.cpp -o CMakeFiles/variational_parameter_test.dir/variational_parameter_test.cpp.s

test/CMakeFiles/variational_parameter_test.dir/variational_parameter_test.cpp.o.requires:
.PHONY : test/CMakeFiles/variational_parameter_test.dir/variational_parameter_test.cpp.o.requires

test/CMakeFiles/variational_parameter_test.dir/variational_parameter_test.cpp.o.provides: test/CMakeFiles/variational_parameter_test.dir/variational_parameter_test.cpp.o.requires
	$(MAKE) -f test/CMakeFiles/variational_parameter_test.dir/build.make test/CMakeFiles/variational_parameter_test.dir/variational_parameter_test.cpp.o.provides.build
.PHONY : test/CMakeFiles/variational_parameter_test.dir/variational_parameter_test.cpp.o.provides

test/CMakeFiles/variational_parameter_test.dir/variational_parameter_test.cpp.o.provides.build: test/CMakeFiles/variational_parameter_test.dir/variational_parameter_test.cpp.o

# Object files for target variational_parameter_test
variational_parameter_test_OBJECTS = \
"CMakeFiles/variational_parameter_test.dir/variational_parameter_test.cpp.o"

# External object files for target variational_parameter_test
variational_parameter_test_EXTERNAL_OBJECTS =

test/variational_parameter_test: test/CMakeFiles/variational_parameter_test.dir/variational_parameter_test.cpp.o
test/variational_parameter_test: test/CMakeFiles/variational_parameter_test.dir/build.make
test/variational_parameter_test: libLinearResponseVariationalBayes.so
test/variational_parameter_test: test/CMakeFiles/variational_parameter_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable variational_parameter_test"
	cd /home/rgiordan/Documents/git_repos/LinearResponseVariationalBayes.cpp/src/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/variational_parameter_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test/CMakeFiles/variational_parameter_test.dir/build: test/variational_parameter_test
.PHONY : test/CMakeFiles/variational_parameter_test.dir/build

test/CMakeFiles/variational_parameter_test.dir/requires: test/CMakeFiles/variational_parameter_test.dir/variational_parameter_test.cpp.o.requires
.PHONY : test/CMakeFiles/variational_parameter_test.dir/requires

test/CMakeFiles/variational_parameter_test.dir/clean:
	cd /home/rgiordan/Documents/git_repos/LinearResponseVariationalBayes.cpp/src/test && $(CMAKE_COMMAND) -P CMakeFiles/variational_parameter_test.dir/cmake_clean.cmake
.PHONY : test/CMakeFiles/variational_parameter_test.dir/clean

test/CMakeFiles/variational_parameter_test.dir/depend:
	cd /home/rgiordan/Documents/git_repos/LinearResponseVariationalBayes.cpp/src && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/rgiordan/Documents/git_repos/LinearResponseVariationalBayes.cpp/src /home/rgiordan/Documents/git_repos/LinearResponseVariationalBayes.cpp/src/test /home/rgiordan/Documents/git_repos/LinearResponseVariationalBayes.cpp/src /home/rgiordan/Documents/git_repos/LinearResponseVariationalBayes.cpp/src/test /home/rgiordan/Documents/git_repos/LinearResponseVariationalBayes.cpp/src/test/CMakeFiles/variational_parameter_test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : test/CMakeFiles/variational_parameter_test.dir/depend

