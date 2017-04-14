cmake_minimum_required(VERSION 2.8.11)
project(PARTICLE_FILTER)

# Build the particle filter project and solution.
set(SRCS src/main.cpp src/particle_filter.cpp)

# Comppiler specific optons
if(MSVC)
    add_definitions("/MP -D_SCL_SECURE_NO_WARNINGS /W4 /WX /wd4005 /wd4127 /wd4481 /wd4482 /wd4505 /wd4512 /wd4714 /wd4800 /wd4189")
else()
    # Use C++11
    set_source_files_properties(${SRCS} PROPERTIES COMPILE_FLAGS -std=c++11)
endif()

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# Create the executable
add_executable(particle_filter ${SRCS})

#if [ ! -f ./src/particle_filter_sol.cpp]; then
#	echo "No solution file."
#else
#	echo "Solution file exists"
#fi

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/particle_filter_sol.cpp")
	set(SRCS src/main.cpp src/particle_filter_sol.cpp)
	set_source_files_properties(${SRCS} PROPERTIES COMPILE_FLAGS -std=c++0x)

	# Create the executable
	add_executable(particle_filter_solution ${SRCS})
endif()
