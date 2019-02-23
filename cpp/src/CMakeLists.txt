SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)


SET(
SRC_LIST "${CMAKE_SOURCE_DIR}/src/cyl_segmenter.hpp" 
"${CMAKE_SOURCE_DIR}/src/cyl_util.hpp" 
"${CMAKE_SOURCE_DIR}/src/main.cpp"
"${CMAKE_SOURCE_DIR}/src/stdafx.cpp"
)

INCLUDE_DIRECTORIES(
${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR} 
"${CMAKE_SOURCE_DIR}/include" 
"${CMAKE_SOURCE_DIR}/include/Eigen/eigen3"
"${CMAKE_SOURCE_DIR}/include/armadillo"
"${CMAKE_SOURCE_DIR}/include/nanoflann"
)

ADD_EXECUTABLE(cylSegmenter ${SRC_LIST})

if (WIN32)
FILE(GLOB_RECURSE MKL_LIB_FILES "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries_*/windows/mkl/lib/intel64_win/mkl_core.lib")
LIST(LENGTH MKL_LIB_FILES PATHS_SIZE)
IF(NOT PATHS_SIZE)
MESSAGE(FATAL_ERROR 
"file mkl_core.lib not found. "
"Path in win64 is usually C:/Program Files (x86)/IntelSWTools/compilers_and_libraries_2019.2.190/windows/mkl/lib/intel64_win/mkl_core.lib ")
ENDIF(NOT PATHS_SIZE)
MESSAGE("Found multiple MKL library paths containing mklcore.lib: ${MKL_LIB_FILES}")
endif (WIN32)


if (UNIX)
FILE(GLOB_RECURSE MKL_LIB_FILES "${MKLROOT}/lib/intel64/libmkl_core.a")
LIST(LENGTH MKL_LIB_FILES PATHS_SIZE)
IF(NOT PATHS_SIZE)
MESSAGE(FATAL_ERROR 
"file libmkl_core.a not found. "
"Path in linux is usually ${MKLROOT}/lib/intel64/libmkl_core.a")
ENDIF(NOT PATHS_SIZE)
MESSAGE("Found multiple MKL library paths containing libmkl_core.a: ${MKL_LIB_FILES}")
endif (UNIX)


LIST(GET MKL_LIB_FILES 0 MKL_LIB_FILE)
GET_FILENAME_COMPONENT(MKL_LIB_PATH ${MKL_LIB_FILE} DIRECTORY)
MESSAGE("Chose MKL library path: ${MKL_LIB_PATH}")

if (WIN32)
	FIND_PROGRAM(MKL_CORE_PATH mkl_core.lib ${MKL_LIB_PATH})
	FIND_PROGRAM(MKL_INTEL_LP64_PATH mkl_intel_lp64.lib ${MKL_LIB_PATH})
	FIND_PROGRAM(MKL_SEQUENTIAL_PATH mkl_sequential.lib ${MKL_LIB_PATH})
endif (WIN32)
if (UNIX)
	FIND_PROGRAM(MKL_CORE_PATH libmkl_core.a ${MKL_LIB_PATH})
	FIND_PROGRAM(MKL_INTEL_LP64_PATH libmkl_intel_lp64.a ${MKL_LIB_PATH})
	FIND_PROGRAM(MKL_SEQUENTIAL_PATH libmkl_sequential.a ${MKL_LIB_PATH})
endif (UNIX)

TARGET_LINK_LIBRARIES(cylSegmenter ${MKL_CORE_PATH} ${MKL_INTEL_LP64_PATH} ${MKL_SEQUENTIAL_PATH})
