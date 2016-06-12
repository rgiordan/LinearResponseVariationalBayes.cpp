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
include CMakeFiles/LinearResponseVariationalBayes.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/LinearResponseVariationalBayes.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/LinearResponseVariationalBayes.dir/flags.make

CMakeFiles/LinearResponseVariationalBayes.dir/exponential_families.cpp.o: CMakeFiles/LinearResponseVariationalBayes.dir/flags.make
CMakeFiles/LinearResponseVariationalBayes.dir/exponential_families.cpp.o: exponential_families.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/rgiordan/Documents/git_repos/LinearResponseVariationalBayes.cpp/src/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/LinearResponseVariationalBayes.dir/exponential_families.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/LinearResponseVariationalBayes.dir/exponential_families.cpp.o -c /home/rgiordan/Documents/git_repos/LinearResponseVariationalBayes.cpp/src/exponential_families.cpp

CMakeFiles/LinearResponseVariationalBayes.dir/exponential_families.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/LinearResponseVariationalBayes.dir/exponential_families.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/rgiordan/Documents/git_repos/LinearResponseVariationalBayes.cpp/src/exponential_families.cpp > CMakeFiles/LinearResponseVariationalBayes.dir/exponential_families.cpp.i

CMakeFiles/LinearResponseVariationalBayes.dir/exponential_families.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/LinearResponseVariationalBayes.dir/exponential_families.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/rgiordan/Documents/git_repos/LinearResponseVariationalBayes.cpp/src/exponential_families.cpp -o CMakeFiles/LinearResponseVariationalBayes.dir/exponential_families.cpp.s

CMakeFiles/LinearResponseVariationalBayes.dir/exponential_families.cpp.o.requires:
.PHONY : CMakeFiles/LinearResponseVariationalBayes.dir/exponential_families.cpp.o.requires

CMakeFiles/LinearResponseVariationalBayes.dir/exponential_families.cpp.o.provides: CMakeFiles/LinearResponseVariationalBayes.dir/exponential_families.cpp.o.requires
	$(MAKE) -f CMakeFiles/LinearResponseVariationalBayes.dir/build.make CMakeFiles/LinearResponseVariationalBayes.dir/exponential_families.cpp.o.provides.build
.PHONY : CMakeFiles/LinearResponseVariationalBayes.dir/exponential_families.cpp.o.provides

CMakeFiles/LinearResponseVariationalBayes.dir/exponential_families.cpp.o.provides.build: CMakeFiles/LinearResponseVariationalBayes.dir/exponential_families.cpp.o

CMakeFiles/LinearResponseVariationalBayes.dir/transform_hessian.cpp.o: CMakeFiles/LinearResponseVariationalBayes.dir/flags.make
CMakeFiles/LinearResponseVariationalBayes.dir/transform_hessian.cpp.o: transform_hessian.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/rgiordan/Documents/git_repos/LinearResponseVariationalBayes.cpp/src/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/LinearResponseVariationalBayes.dir/transform_hessian.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/LinearResponseVariationalBayes.dir/transform_hessian.cpp.o -c /home/rgiordan/Documents/git_repos/LinearResponseVariationalBayes.cpp/src/transform_hessian.cpp

CMakeFiles/LinearResponseVariationalBayes.dir/transform_hessian.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/LinearResponseVariationalBayes.dir/transform_hessian.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/rgiordan/Documents/git_repos/LinearResponseVariationalBayes.cpp/src/transform_hessian.cpp > CMakeFiles/LinearResponseVariationalBayes.dir/transform_hessian.cpp.i

CMakeFiles/LinearResponseVariationalBayes.dir/transform_hessian.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/LinearResponseVariationalBayes.dir/transform_hessian.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/rgiordan/Documents/git_repos/LinearResponseVariationalBayes.cpp/src/transform_hessian.cpp -o CMakeFiles/LinearResponseVariationalBayes.dir/transform_hessian.cpp.s

CMakeFiles/LinearResponseVariationalBayes.dir/transform_hessian.cpp.o.requires:
.PHONY : CMakeFiles/LinearResponseVariationalBayes.dir/transform_hessian.cpp.o.requires

CMakeFiles/LinearResponseVariationalBayes.dir/transform_hessian.cpp.o.provides: CMakeFiles/LinearResponseVariationalBayes.dir/transform_hessian.cpp.o.requires
	$(MAKE) -f CMakeFiles/LinearResponseVariationalBayes.dir/build.make CMakeFiles/LinearResponseVariationalBayes.dir/transform_hessian.cpp.o.provides.build
.PHONY : CMakeFiles/LinearResponseVariationalBayes.dir/transform_hessian.cpp.o.provides

CMakeFiles/LinearResponseVariationalBayes.dir/transform_hessian.cpp.o.provides.build: CMakeFiles/LinearResponseVariationalBayes.dir/transform_hessian.cpp.o

CMakeFiles/LinearResponseVariationalBayes.dir/variational_parameters.cpp.o: CMakeFiles/LinearResponseVariationalBayes.dir/flags.make
CMakeFiles/LinearResponseVariationalBayes.dir/variational_parameters.cpp.o: variational_parameters.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/rgiordan/Documents/git_repos/LinearResponseVariationalBayes.cpp/src/CMakeFiles $(CMAKE_PROGRESS_3)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/LinearResponseVariationalBayes.dir/variational_parameters.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/LinearResponseVariationalBayes.dir/variational_parameters.cpp.o -c /home/rgiordan/Documents/git_repos/LinearResponseVariationalBayes.cpp/src/variational_parameters.cpp

CMakeFiles/LinearResponseVariationalBayes.dir/variational_parameters.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/LinearResponseVariationalBayes.dir/variational_parameters.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/rgiordan/Documents/git_repos/LinearResponseVariationalBayes.cpp/src/variational_parameters.cpp > CMakeFiles/LinearResponseVariationalBayes.dir/variational_parameters.cpp.i

CMakeFiles/LinearResponseVariationalBayes.dir/variational_parameters.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/LinearResponseVariationalBayes.dir/variational_parameters.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/rgiordan/Documents/git_repos/LinearResponseVariationalBayes.cpp/src/variational_parameters.cpp -o CMakeFiles/LinearResponseVariationalBayes.dir/variational_parameters.cpp.s

CMakeFiles/LinearResponseVariationalBayes.dir/variational_parameters.cpp.o.requires:
.PHONY : CMakeFiles/LinearResponseVariationalBayes.dir/variational_parameters.cpp.o.requires

CMakeFiles/LinearResponseVariationalBayes.dir/variational_parameters.cpp.o.provides: CMakeFiles/LinearResponseVariationalBayes.dir/variational_parameters.cpp.o.requires
	$(MAKE) -f CMakeFiles/LinearResponseVariationalBayes.dir/build.make CMakeFiles/LinearResponseVariationalBayes.dir/variational_parameters.cpp.o.provides.build
.PHONY : CMakeFiles/LinearResponseVariationalBayes.dir/variational_parameters.cpp.o.provides

CMakeFiles/LinearResponseVariationalBayes.dir/variational_parameters.cpp.o.provides.build: CMakeFiles/LinearResponseVariationalBayes.dir/variational_parameters.cpp.o

# Object files for target LinearResponseVariationalBayes
LinearResponseVariationalBayes_OBJECTS = \
"CMakeFiles/LinearResponseVariationalBayes.dir/exponential_families.cpp.o" \
"CMakeFiles/LinearResponseVariationalBayes.dir/transform_hessian.cpp.o" \
"CMakeFiles/LinearResponseVariationalBayes.dir/variational_parameters.cpp.o"

# External object files for target LinearResponseVariationalBayes
LinearResponseVariationalBayes_EXTERNAL_OBJECTS =

libLinearResponseVariationalBayes.so: CMakeFiles/LinearResponseVariationalBayes.dir/exponential_families.cpp.o
libLinearResponseVariationalBayes.so: CMakeFiles/LinearResponseVariationalBayes.dir/transform_hessian.cpp.o
libLinearResponseVariationalBayes.so: CMakeFiles/LinearResponseVariationalBayes.dir/variational_parameters.cpp.o
libLinearResponseVariationalBayes.so: CMakeFiles/LinearResponseVariationalBayes.dir/build.make
libLinearResponseVariationalBayes.so: CMakeFiles/LinearResponseVariationalBayes.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX shared library libLinearResponseVariationalBayes.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/LinearResponseVariationalBayes.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/LinearResponseVariationalBayes.dir/build: libLinearResponseVariationalBayes.so
.PHONY : CMakeFiles/LinearResponseVariationalBayes.dir/build

CMakeFiles/LinearResponseVariationalBayes.dir/requires: CMakeFiles/LinearResponseVariationalBayes.dir/exponential_families.cpp.o.requires
CMakeFiles/LinearResponseVariationalBayes.dir/requires: CMakeFiles/LinearResponseVariationalBayes.dir/transform_hessian.cpp.o.requires
CMakeFiles/LinearResponseVariationalBayes.dir/requires: CMakeFiles/LinearResponseVariationalBayes.dir/variational_parameters.cpp.o.requires
.PHONY : CMakeFiles/LinearResponseVariationalBayes.dir/requires

CMakeFiles/LinearResponseVariationalBayes.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/LinearResponseVariationalBayes.dir/cmake_clean.cmake
.PHONY : CMakeFiles/LinearResponseVariationalBayes.dir/clean

CMakeFiles/LinearResponseVariationalBayes.dir/depend:
	cd /home/rgiordan/Documents/git_repos/LinearResponseVariationalBayes.cpp/src && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/rgiordan/Documents/git_repos/LinearResponseVariationalBayes.cpp/src /home/rgiordan/Documents/git_repos/LinearResponseVariationalBayes.cpp/src /home/rgiordan/Documents/git_repos/LinearResponseVariationalBayes.cpp/src /home/rgiordan/Documents/git_repos/LinearResponseVariationalBayes.cpp/src /home/rgiordan/Documents/git_repos/LinearResponseVariationalBayes.cpp/src/CMakeFiles/LinearResponseVariationalBayes.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/LinearResponseVariationalBayes.dir/depend

