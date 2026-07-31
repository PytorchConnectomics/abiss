cmake_minimum_required(VERSION 3.8)
project(ABISS)
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
set(optionalLibs)


set(CMAKE_CXX_STANDARD 20)

set(ABISS_ARCH "" CACHE STRING
    "Architecture passed to -march (empty preserves inherited compiler flags)")
if(ABISS_ARCH)
    if(NOT ABISS_ARCH MATCHES "^[A-Za-z0-9_.+-]+$")
        message(FATAL_ERROR "ABISS_ARCH must be a single architecture name")
    endif()
    string(REGEX REPLACE "(^|[ \t])-march=[^ \t]+" " " CMAKE_CXX_FLAGS
        "${CMAKE_CXX_FLAGS}")
    string(REGEX REPLACE "(^|[ \t])-mtune=[^ \t]+" " " CMAKE_CXX_FLAGS
        "${CMAKE_CXX_FLAGS}")
    string(STRIP "${CMAKE_CXX_FLAGS}" CMAKE_CXX_FLAGS)
    string(APPEND CMAKE_CXX_FLAGS " -march=${ABISS_ARCH}")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>")
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fopenmp")
set(CMAKE_CXX_FLAGS_DEBUG "-Og -g -fopenmp")

set(THREADS_PREFER_PTHREAD_FLAG ON)

find_package(Boost COMPONENTS iostreams system REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)
find_package(TBB REQUIRED)
find_package(absl)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(absl_FOUND FALSE)
    message(STATUS "Debug mode: forcing std::unordered_map")
endif()

set(ABISS_ALLOCATOR "system" CACHE STRING
    "Allocator implementation: system, mimalloc, or jemalloc")
set_property(CACHE ABISS_ALLOCATOR PROPERTY STRINGS system mimalloc jemalloc)
if(NOT ABISS_ALLOCATOR MATCHES "^(system|mimalloc|jemalloc)$")
    message(FATAL_ERROR
        "ABISS_ALLOCATOR must be one of: system, mimalloc, jemalloc")
endif()

if(ABISS_ALLOCATOR STREQUAL "system")
    message(STATUS "Use the system allocator.")
elseif(ABISS_ALLOCATOR STREQUAL "mimalloc")
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(MIMALLOC REQUIRED IMPORTED_TARGET mimalloc)
    add_definitions(-DUSE_MIMALLOC)
    list(APPEND optionalLibs PkgConfig::MIMALLOC)
    message(STATUS "Use mimalloc as the default allocator.")
elseif(ABISS_ALLOCATOR STREQUAL "jemalloc")
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(JEMALLOC REQUIRED IMPORTED_TARGET jemalloc)
    list(APPEND optionalLibs PkgConfig::JEMALLOC)
    message(STATUS "Use jemalloc as the default allocator.")
endif()

if(absl_FOUND)
    add_definitions(-DUSE_ABSL_HASHMAP)
    set(optionalLibs ${optionalLibs}
        absl::flat_hash_map
        absl::flat_hash_set
        )
    message(STATUS "Use abseil::flat_hash_map and absl::flat_hash_set")
endif(absl_FOUND)

link_libraries(TBB::tbb ${Boost_LIBRARIES} ${optionalLibs})

if(EXTRACT_SIZE)
    add_definitions(-DEXTRACT_SIZE)
endif(EXTRACT_SIZE)

if(EXTRACT_BBOX)
    add_definitions(-DEXTRACT_BBOX)
endif(EXTRACT_BBOX)

if(EXTRACT_COM)
    add_definitions(-DEXTRACT_COM)
endif(EXTRACT_COM)


if(MST_EDGE)
    add_definitions(-DMST_EDGE)
endif(MST_EDGE)

if(DOUBLE)
    add_definitions(-DDOUBLE)
endif(DOUBLE)

add_executable(meme
    src/seg/merge_edges_ME.cpp)
add_executable(acme
    src/seg/atomic_chunk_ME.cpp)
add_executable(assort
    src/seg/assort.cpp)
add_executable(split_remap
    src/seg/split_remap.cpp)
add_executable(match_chunks
    src/seg/match_chunks.cpp)
add_executable(reduce_chunk
    src/seg/reduce_chunk.cpp)
add_executable(size_map
    src/seg/size_map.cpp)
add_executable(evaluate
    src/seg/eval.cpp)
add_executable(ws
    src/ws/atomic_chunk.cpp)
add_executable(ws2
    src/ws/merge_chunks.cpp)
add_executable(ws3
    src/ws/remap.cpp)
add_executable(agg
    src/agg/mean_aggl.cpp)
add_executable(agg_nonoverlap
    src/agg/mean_aggl.cpp)
add_executable(agg_overlap
    src/agg/mean_aggl.cpp)
add_executable(agg_extra
    src/agg/mean_aggl.cpp)
add_executable(accs
    src/seg/contact_surface.cpp)
add_executable(mecs
    src/seg/merge_cs.cpp)
#target_compile_definitions(agg PRIVATE FINAL PRIVATE EXTRA)
#target_compile_definitions(agg_overlap PRIVATE OVERLAPPED PRIVATE FINAL PRIVATE EXTRA)
#target_compile_definitions(agg_extra PRIVATE EXTRA)
target_compile_definitions(agg PRIVATE FINAL)
target_compile_definitions(agg_overlap PRIVATE OVERLAPPED PRIVATE FINAL)
