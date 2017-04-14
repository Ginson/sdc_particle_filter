cmake_minimum_required(VERSION 2.8.11)
project(PARTICLE_FILTER)

# Comppiler specific optons
if(MSVC)
    add_definitions("/MP -D_SCL_SECURE_NO_WARNINGS /W4 /WX /wd4005 /wd4127 /wd4481 /wd4482 /wd4505 /wd4512 /wd4714 /wd4800 /wd4189")
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data DESTINATION ${RUNTIME_OUTPUT_DIRECTORY}/Debug)
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data DESTINATION ${RUNTIME_OUTPUT_DIRECTORY}/Release)
else()
    add_definitions(-std=c++0x)
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endif()

# Build the particle filter project and solution.
# Use C++11
set(SRCS src/main.cpp src/particle_filter.cpp)
set_source_files_properties(${SRCS} PROPERTIES COMPILE_FLAGS -std=c++0x)

# Create the executable
add_executable(particle_filter ${SRCS})

# Use C++11

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






