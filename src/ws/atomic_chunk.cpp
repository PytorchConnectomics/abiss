//#pragma once
#include "agglomeration.hpp"
#include "region_graph.hpp"
#include "basic_watershed.hpp"
#include "types.hpp"
#include "utils.hpp"
#include "mmap_array.hpp"

#include <memory>
#include <type_traits>

#include <iostream>
#include <fstream>
#include <cstddef>
#include <cstdint>
#include <cassert>
#include <cctype>
#include <vector>
#include <algorithm>
#include <tuple>
#include <string>
#include <vector>
#include <chrono>
#include <ctime>
#include <boost/format.hpp>

// Watershed packs traversal flags into internal ids, reserving high_bit; the
// padded voxel count must stay below 2^31 for ws or 2^63 for ws64. The wider
// ids increase segmentation and region-graph memory, so ws uses uint32 by
// default. On-disk segmentation width is selected independently by --seg-dtype;
// boundary segmentation and graph ids remain uint64.
#ifdef WS_INTERNAL_SEG64
using internal_seg_t = uint64_t;
#else
using internal_seg_t = uint32_t;
#endif

template< typename IT, typename OT, typename F>
region_graph<OT, F> relabel_region_graph(const region_graph<IT, F>& rg, OT offset)
{
    region_graph<OT, F> new_rg;
    for (auto & e : rg) {
        new_rg.emplace_back(std::get<0>(e), static_cast<OT>(std::get<1>(e)) + offset, static_cast<OT>(std::get<2>(e)) + offset);
    }
    return new_rg;
}

int main(int argc, char* argv[]) try
{
    if (argc < 8) {
        std::cerr << "Usage: ws param aff high low size dust tag [merge_func] [thresholds...] [--seg-dtype=uint32|uint64]" << std::endl;
        return 2;
    }
    bool output_u32 = false, dtype_seen = false;
    int retained = 8;
    for (int i = 8; i < argc; ++i) {
        std::string token(argv[i]);
        if (token.rfind("--seg-dtype=", 0) == 0) {
            if (dtype_seen || (token != "--seg-dtype=uint32" && token != "--seg-dtype=uint64")) {
                std::cerr << "Invalid or duplicate --seg-dtype token: " << token << std::endl;
                return 2;
            }
            dtype_seen = true;
            output_u32 = token == "--seg-dtype=uint32";
        } else {
            argv[retained++] = argv[i];
        }
    }
    argc = retained;

    size_t xdim = 0, ydim = 0, zdim = 0;
    int flag;
    seg_t offset = 0;
    std::ifstream param_file(argv[1]);
    std::string ht(argv[3]);
    std::string lt(argv[4]);
    std::string st(argv[5]);
    std::string dt(argv[6]);
    const char * tag = argv[7];

    // Parse optional merge function and merge thresholds from argv[8..N].
    // When multiple thresholds are given, watershed + region graph are
    // computed once and the merge step is repeated for each threshold,
    // writing indexed output files.
    //
    // CLI format: ws param aff high low size dust tag [merge_func] [thresholds...]
    // merge_func: "max" (default), "mean", or "pNN" (e.g. "p75", "p90")
    std::vector<aff_t> merge_thresholds;
    auto high_threshold = read_float<aff_t>(ht);
    auto low_threshold = read_float<aff_t>(lt);
    auto size_threshold = read_int(st);
    auto dust_threshold = read_int(dt);

    EdgeScoreConfig score_cfg;  // default: MAX
    int merge_thresh_start = 8;

    // If argv[8] starts with a letter, it's a merge function spec.
    // Merge thresholds start with a digit, dot, or minus.
    if (argc > 8 && std::isalpha(static_cast<unsigned char>(argv[8][0]))) {
        std::string merge_func_str(argv[8]);
        score_cfg = parse_edge_score_config(merge_func_str);
        merge_thresh_start = 9;
        std::cout << "merge function: " << merge_func_str << std::endl;
    }

    if (argc > merge_thresh_start) {
        for (int i = merge_thresh_start; i < argc; i++) {
            std::string ms(argv[i]);
            merge_thresholds.push_back(read_float<aff_t>(ms));
        }
    } else {
        merge_thresholds.push_back(low_threshold);
    }

    std::cout << "thresholds: " << ht << " " << lt << " " << st << " " << dt
              << " merge=[";
    for (size_t i = 0; i < merge_thresholds.size(); i++) {
        if (i > 0) std::cout << ",";
        std::cout << merge_thresholds[i];
    }
    std::cout << "]" << std::endl;

    size_t chunk_size;
    if (!(param_file >> xdim >> ydim >> zdim)
        || !watershed_size_valid<internal_seg_t>(xdim, ydim, zdim, chunk_size)) {
        std::cerr << "Invalid chunk size for ws" << sizeof(internal_seg_t)*8
                  << ": dimension product overflows or exceeds watershed index limit" << std::endl;
        return 2;
    }
    std::cout << "Chunk size check passed: " << chunk_size << std::endl;
    if (xdim < 3 || ydim < 3 || zdim < 3) {
        std::cerr << "Each dimension must include an interior and two halo voxels" << std::endl;
        return 2;
    }
    std::cout << xdim << " " << ydim << " " << zdim << std::endl;

#ifdef USE_MIMALLOC
    size_t huge_pages = xdim * ydim * zdim * 4 * 3 * 4 / 1024 / 1024 / 1024 + 1;
    auto mi_ret = mi_reserve_huge_os_pages_interleave(huge_pages, 0, 0);
    if (mi_ret == ENOMEM) {
       std::cout << "failed to reserve 1GB huge pages" << std::endl;
    }
#endif

    std::array<bool,6> flags({true,true,true,true,true,true});
    for (size_t i = 0; i != 6; i++) {
        param_file >> flag;
        flags[i] = (flag > 0);
        if (flags[i]) {
            std::cout << "real boundary: " << i << std::endl;
        }
    }
    param_file >> offset;
    std::cout << "supervoxel id offset:" << offset << std::endl;


    clock_t begin = clock();
    std::array<size_t, 4> aff_dim({xdim,ydim,zdim,3});
    MMArray<aff_t, 4> aff_data(argv[2], aff_dim);
    affinity_graph_ptr<aff_t> aff = aff_data.data_ptr();
    //    read_affinity_graph<float>(argv[2],
    //                               xdim, ydim, zdim);
    //                               //2050, 2050, 258);
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    std::cout << "loaded affinity map in " << elapsed_secs << " seconds" << std::endl;

    volume_ptr<internal_seg_t> seg;
    std::vector<std::size_t> counts;

    begin = clock();
    memory_marker("watershed: begin");
    if (bfs_fits_u32(chunk_size))
        std::tie(seg, counts) = watershed<internal_seg_t, uint32_t>(aff, low_threshold, high_threshold, flags);
    else
        std::tie(seg, counts) = watershed<internal_seg_t, std::ptrdiff_t>(aff, low_threshold, high_threshold, flags);
    memory_marker("watershed: end");
    end = clock();
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    std::cout << "finished watershed in " << elapsed_secs << " seconds" << std::endl;
    begin = clock();
    auto rg = get_region_graph(aff, seg , counts.size()-1, low_threshold, flags, score_cfg);
    end = clock();
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    std::cout << "finished region graph in " << elapsed_secs << " seconds" << std::endl;

    if (merge_thresholds.size() > 1)
        std::cout << "Multi-threshold mode: " << merge_thresholds.size()
                  << " merge thresholds" << std::endl;
    for (size_t mi = 0; mi < merge_thresholds.size(); ++mi) {
        const clock_t mt_begin = clock();
        begin = clock();
        memory_marker("merge: counts copy begin");
        auto merged_counts = counts;
        auto merged = compute_merge(rg, merged_counts,
                                    std::make_pair(size_threshold, merge_thresholds[mi]),
                                    dust_threshold);
        elapsed_secs = double(clock() - begin) / CLOCKS_PER_SEC;
        std::cout << "finished agglomeration in " << elapsed_secs << " seconds" << std::endl;
        const auto& lut = merged.first;
        const size_t max_id = merged_counts.size()-1;
        // Check compact labels, not the original watershed LUT length.
        if (output_u32 && (offset > UINT32_MAX || max_id > UINT32_MAX - offset)) {
            std::cerr << "uint32 segmentation overflow at threshold " << mi
                      << ": offset=" << offset << " max_id=" << max_id << std::endl;
            return 3;
        }
        std::string out_tag = merge_thresholds.size() == 1 ? std::string(tag)
                              : str(boost::format("%1%_%2%") % tag % mi);
        auto transform = [&lut, offset](internal_seg_t id) -> seg_t {
            const seg_t v = lut[id];
            return v == 0 ? 0 : v + offset;
        };
        memory_marker("write: begin");
        auto relabeled_rg = relabel_region_graph(merged.second, offset);
        free_container(merged.second);
        auto c = write_counts(merged_counts, offset, out_tag.c_str());
        free_container(merged_counts);
        auto d = write_vector(str(boost::format("dend_%1%.data") % out_tag), relabeled_rg);
        free_container(relabeled_rg);
        const auto filename = str(boost::format("seg_%1%.data") % out_tag);
        begin = clock();
        if (output_u32) write_volume<uint32_t>(filename, seg, transform);
        else write_volume<seg_t>(filename, seg, transform);
        memory_marker("write: volume end / faces begin");
        write_chunk_boundaries(seg, aff, flags, out_tag.c_str(), transform);
        std::vector<size_t> meta({xdim,ydim,zdim,c,d,0});
        write_vector(str(boost::format("meta_%1%.data") % out_tag), meta);
        memory_marker("write: end");
        std::cout << "num of sv:" << c << std::endl;
        std::cout << "size of rg:" << d << std::endl;
        elapsed_secs = double(clock() - begin) / CLOCKS_PER_SEC;
        std::cout << "finished writing in " << elapsed_secs << " seconds" << std::endl;
        if (merge_thresholds.size() > 1) {
            const double mt_secs = double(clock() - mt_begin) / CLOCKS_PER_SEC;
            std::cout << "merge threshold " << mi << " (" << merge_thresholds[mi]
                      << "): sv=" << c << " rg=" << d
                      << " in " << mt_secs << " seconds" << std::endl;
        }
    }

    return 0;
}
catch (const std::exception& error)
{
    // Exit 4 reports runtime failures, including streamed output I/O errors.
    std::cerr << "ws runtime error: " << error.what() << std::endl;
    return 4;
}
