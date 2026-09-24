#pragma once

#include "types.hpp"
#include "edge_score.hpp"
#include "utils.hpp"

#include <cstddef>
#include <iostream>
#include <vector>

template<typename F, EdgeScoreMode Mode>
struct EdgeAccumulator {
    F value = F(0);
    size_t count = 0;
    bool empty() const { return count == 0; }
    void push_back(F a) {
        if constexpr (Mode == EdgeScoreMode::MAX) {
            if (!count || value < a) value = a;
        } else {
            value = value + a;
        }
        ++count;
    }
    F score(const EdgeScoreConfig&) {
        if constexpr (Mode == EdgeScoreMode::MAX) return value;
        else return value / static_cast<F>(count);
    }
};

template<typename F>
struct EdgeAccumulator<F, EdgeScoreMode::PERCENTILE> : std::vector<F> {
    F score(const EdgeScoreConfig& cfg) { return compute_edge_score(*this, cfg); }
};

template<EdgeScoreMode Mode, typename ID, typename F, typename L>
inline region_graph<ID,F>
get_region_graph_impl( const affinity_graph_ptr<F>& aff_ptr,
                  const volume_ptr<ID> seg_ptr,
                  std::size_t max_segid, const L& lowv, const std::array<bool,6> & boundary_flags,
                  const EdgeScoreConfig& score_cfg = EdgeScoreConfig())
{
    using affinity_t = F;
    using id_pair = std::pair<ID,ID>;
    affinity_t low  = static_cast<affinity_t>(lowv);

    std::ptrdiff_t xdim = aff_ptr->shape()[0];
    std::ptrdiff_t ydim = aff_ptr->shape()[1];
    std::ptrdiff_t zdim = aff_ptr->shape()[2];

    volume<ID>& seg = *seg_ptr;
    affinity_graph<F> aff = *aff_ptr;

    region_graph<ID,F> rg;

    std::vector<id_pair> pairs;

    memory_marker("graph: fill begin");
    std::vector<MapContainer<ID, EdgeAccumulator<F, Mode>>> edges(max_segid);
    for (auto & h : edges) {
        h.reserve(10);
    }

    for ( std::ptrdiff_t z = 1; z < zdim - 1; ++z )
        for ( std::ptrdiff_t y = 1; y < ydim - 1; ++y )
            for ( std::ptrdiff_t x = 1; x < xdim - 1; ++x )
            {
                if ( (x > boundary_flags[0]) && seg[x][y][z] && seg[x-1][y][z] && seg[x][y][z] != seg[x-1][y][z])
                {
                    auto p = std::minmax(seg[x][y][z], seg[x-1][y][z]);
                    auto& vec = edges[p.first][p.second];
                    if (vec.empty()) {
                        pairs.push_back(p);
                    }
                    vec.push_back(aff[x][y][z][0]);
                }
                if ( (y > boundary_flags[1]) && seg[x][y][z] && seg[x][y-1][z] && seg[x][y][z] != seg[x][y-1][z])
                {
                    auto p = std::minmax(seg[x][y][z], seg[x][y-1][z]);
                    auto& vec = edges[p.first][p.second];
                    if (vec.empty()) {
                        pairs.push_back(p);
                    }
                    vec.push_back(aff[x][y][z][1]);
                }
                if ( (z > boundary_flags[2]) && seg[x][y][z] && seg[x][y][z-1] && seg[x][y][z] != seg[x][y][z-1])
                {
                    auto p = std::minmax(seg[x][y][z], seg[x][y][z-1]);
                    auto& vec = edges[p.first][p.second];
                    if (vec.empty()) {
                        pairs.push_back(p);
                    }
                    vec.push_back(aff[x][y][z][2]);
                }
            }

    memory_marker("graph: fill end / score begin");
    for ( const auto& p : pairs)
    {
        auto& affs = edges[p.first][p.second];
        F score = affs.score(score_cfg);
        rg.emplace_back(score, p.first, p.second);
    }

    memory_marker("graph: score end");
    free_container(edges);
    free_container(pairs);
    std::cout << "Region graph size: " << rg.size() << std::endl;

    memory_marker("graph: sort begin");
    std::stable_sort(std::begin(rg), std::end(rg), [](auto & a, auto & b) { return std::get<0>(a) > std::get<0>(b); });
    memory_marker("graph: sort end");

    std::cout << "Sorted" << std::endl;
    return rg;
}

template<typename ID, typename F, typename L>
inline region_graph<ID,F>
get_region_graph(const affinity_graph_ptr<F>& aff, const volume_ptr<ID>& seg,
                 size_t max_segid, const L& low, const std::array<bool,6>& flags,
                 const EdgeScoreConfig& cfg = EdgeScoreConfig())
{
    switch (cfg.mode) {
        case EdgeScoreMode::MAX:
            return get_region_graph_impl<EdgeScoreMode::MAX>(aff, seg, max_segid, low, flags, cfg);
        case EdgeScoreMode::MEAN:
            return get_region_graph_impl<EdgeScoreMode::MEAN>(aff, seg, max_segid, low, flags, cfg);
        case EdgeScoreMode::PERCENTILE:
            return get_region_graph_impl<EdgeScoreMode::PERCENTILE>(aff, seg, max_segid, low, flags, cfg);
    }
    throw std::invalid_argument("Unknown edge score mode");
}
