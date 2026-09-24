#pragma once

#include "types.hpp"

#include <cstdio>
#include <cassert>
#include <fstream>
#include <type_traits>
#include <sstream>
#include <limits>
#include <stdexcept>
#include <boost/format.hpp>
#include <boost/iostreams/device/mapped_file.hpp>

namespace bio = boost::iostreams;

inline void memory_marker(const char* label)
{
    std::ifstream status("/proc/self/status");
    std::string line, key;
    double anon = 0, file = 0, hwm = 0, kb;
    while (std::getline(status, line)) {
        std::istringstream fields(line);
        if (!(fields >> key >> kb)) continue;
        if (key == "RssAnon:") anon = kb;
        if (key == "RssFile:") file = kb;
        if (key == "VmHWM:") hwm = kb;
    }
    std::cout << "[mem] " << label << ": anon_gb=" << anon / 1048576
              << " file_gb=" << file / 1048576 << " hwm_gb=" << hwm / 1048576
              << std::endl;
}

template<typename ID>
inline bool watershed_size_valid(size_t x, size_t y, size_t z, size_t& size)
{
    return !__builtin_mul_overflow(x, y, &size)
        && !__builtin_mul_overflow(size, z, &size)
        && size < watershed_traits<ID>::high_bit
        && size <= static_cast<size_t>(PTRDIFF_MAX);
}

inline bool bfs_fits_u32(size_t size)
{
    return size == 0 || size - 1 <= UINT32_MAX;
}

template < typename T >
inline void free_container(T& p_container)
{
    T empty;
    using std::swap;
    swap(p_container, empty);
}

template < typename T >
inline bool read_from_file( const std::string& fname, T* data, std::size_t n )
{
    FILE* f = std::fopen(fname.c_str(), "rbXS");
    if ( !f ) return false;

    std::size_t nread = std::fread(data, sizeof(T), n, f);
    std::fclose(f);

    return nread == n;
}

template < typename T >
inline bool
write_to_file( const std::string& fname,
               const T* data, std::size_t n )
{
    std::ofstream f(fname.c_str(), (std::ios::out | std::ios::binary) );
    assert(f);

    f.write( reinterpret_cast<const char*>(data), n * sizeof(T));
    assert(!f.bad());
    f.close();
    return true;
}


template < typename T >
inline volume_ptr<T>
read_volume( const std::string& fname, std::size_t wsize )
{
    volume_ptr<T> vol(new volume<T>
                      (boost::extents[wsize][wsize][wsize],
                       boost::fortran_storage_order()));

    if ( !read_from_file(fname, vol->data(), wsize*wsize*wsize) ) throw 0;
    return vol;
}

template< typename ID, typename F >
inline bool write_region_graph( const std::string& fname,
                                const region_graph<ID,F>& rg )
{
    std::ofstream f(fname.c_str(), (std::ios::out | std::ios::binary) );
    if ( !f ) return false;

    F* data = new F[rg.size() * 3];

    std::size_t idx = 0;

    for ( const auto& e: rg )
    {
        data[idx++] = static_cast<F>(std::get<1>(e));
        data[idx++] = static_cast<F>(std::get<2>(e));
        data[idx++] = static_cast<F>(std::get<0>(e));
    }

    f.write( reinterpret_cast<char*>(data), rg.size() * 3 * sizeof(F));

    assert(!f.bad());

    delete [] data;


    f.close();
    return true;
}

template< typename ID >
inline std::tuple<volume_ptr<ID>, std::vector<std::size_t>>
    get_dummy_segmentation( std::size_t xdim,
                            std::size_t ydim,
                            std::size_t zdim )
{
    std::tuple<volume_ptr<ID>, std::vector<std::size_t>> result
        ( volume_ptr<ID>( new volume<ID>(boost::extents[xdim][ydim][zdim],
                                         boost::fortran_storage_order())),
          std::vector<std::size_t>(xdim*ydim*zdim+1));

    volume<ID>& seg = *(std::get<0>(result));
    auto& counts = std::get<1>(result);

    std::fill_n(counts.begin(), xdim*ydim*zdim*1, 1);
    counts[0] = 0;

    for ( ID i = 0; i < xdim*ydim*zdim; ++i )
    {
        seg.data()[i] = i+1;
    }

    return result;
}

template <typename T>
size_t write_vector(const std::string & fn, std::vector<T> & v)
{
    if (v.empty()) {
        std::ofstream fs;
        fs.open(fn);
        fs.close();
        return 0;
    }
    bio::mapped_file_params f_param;
    bio::mapped_file_sink f;
    size_t bytes = sizeof(T)*v.size();
    f_param.path = fn;
    f_param.new_file_size = bytes;
    f.open(f_param);
    assert(f.is_open());
    memcpy(f.data(), v.data(), bytes);
    f.close();
    return v.size();
}

template <typename T>
size_t write_counts(std::vector<size_t> & counts, T & offset, const char * tag)
{
    std::vector<std::pair<T, size_t> > output;
    for (T i = 1; i != counts.size(); i++) {
        output.emplace_back(i+offset, counts[i]);
    }
    return write_vector(str(boost::format("counts_%1%.data") % tag), output);
}

template <typename K, typename V>
size_t write_remap(const MapContainer<K, V> & map, const char * tag)
{
    std::vector<std::pair<K, V> > output;
    for (const auto & kv : map) {
        output.push_back(kv);
    }
    return write_vector(str(boost::format("remap_%1%.data") % tag), output);
}

template<typename T, size_t N>
bool write_multi_array(const std::string & fn, boost::multi_array<T,N> slice){
    bio::mapped_file_params f_param;
    bio::mapped_file_sink f;
    size_t bytes = sizeof(T)*slice.num_elements();
    f_param.path = fn;
    f_param.new_file_size = bytes;
    f.open(f_param);
    assert(f.is_open());
    memcpy(f.data(), slice.data(), bytes);
    f.close();
    return true;
}

template<typename Out, typename ID, typename Transform>
void write_volume(const std::string& fname, const volume_ptr<ID>& seg_ptr,
                  Transform transform)
{
    const auto& seg = *seg_ptr;
    const auto shape = seg.shape();
    const std::string tmp = fname + ".tmp";
    try {
        std::ofstream out;
        out.exceptions(std::ios::failbit | std::ios::badbit);
        out.open(tmp, std::ios::binary | std::ios::trunc);
        std::vector<Out> plane((shape[0]-2) * (shape[1]-2));
        for (size_t z = 1; z < shape[2]-1; ++z) {
            size_t i = 0;
            for (size_t y = 1; y < shape[1]-1; ++y)
                for (size_t x = 1; x < shape[0]-1; ++x)
                    plane[i++] = static_cast<Out>(transform(seg[x][y][z]));
            out.write(reinterpret_cast<const char*>(plane.data()), plane.size() * sizeof(Out));
        }
        out.close();
        if (std::rename(tmp.c_str(), fname.c_str()) != 0)
            throw std::runtime_error("Cannot rename segmentation: " + fname);
    } catch (...) {
        std::remove(tmp.c_str());
        throw;
    }
}

template<typename ID, typename F, typename Transform>
void write_chunk_boundaries(const volume_ptr<ID>& seg_ptr,
                            const affinity_graph_ptr<F>& aff_ptr,
                            const std::array<bool,6>& boundary_flags,
                            const char* tag, Transform transform)
{
    using range = boost::multi_array_types::index_range;
    auto& aff = *aff_ptr;
    const auto& seg = *seg_ptr;
    const auto shape = seg.shape();
    if (!boundary_flags[0]) {
        write_multi_array(str(boost::format("aff_i_0_%1%.data") % tag), boost::multi_array<F,2>(aff[boost::indices[1][range(1,shape[1]-1)][range(1,shape[2]-1)][0]], boost::fortran_storage_order()));
    }
    if (!boundary_flags[1]) {
        write_multi_array(str(boost::format("aff_i_1_%1%.data") % tag), boost::multi_array<F,2>(aff[boost::indices[range(1,shape[0]-1)][1][range(1,shape[2]-1)][1]], boost::fortran_storage_order()));
    }
    if (!boundary_flags[2]) {
        write_multi_array(str(boost::format("aff_i_2_%1%.data") % tag), boost::multi_array<F,2>(aff[boost::indices[range(1,shape[0]-1)][range(1,shape[1]-1)][1][2]], boost::fortran_storage_order()));
    }
    for (size_t face = 0; face < 6; ++face) {
        if (boundary_flags[face]) continue;
        const size_t axis = face % 3;
        for (int inner = 0; inner < 2; ++inner) {
            const size_t pos = face < 3 ? inner : shape[axis]-1-inner;
            std::vector<seg_t> values;
            size_t extent[3] = {shape[0]-2, shape[1]-2, shape[2]-2};
            extent[axis] = 1;
            values.reserve(extent[0] * extent[1] * extent[2]);
            for (size_t z = 0; z < extent[2]; ++z)
                for (size_t y = 0; y < extent[1]; ++y)
                    for (size_t x = 0; x < extent[0]; ++x) {
                        size_t p[3] = {x+1, y+1, z+1};
                        p[axis] = pos;
                        values.push_back(transform(seg[p[0]][p[1]][p[2]]));
                    }
            write_vector(str(boost::format("seg_%1%_%2%_%3%.data")
                             % (inner ? "i" : "o") % face % tag), values);
        }
    }
}

template< class C >
struct is_numeric:
    std::integral_constant<bool,
                           std::is_integral<C>::value ||
                           std::is_floating_point<C>::value> {};
