#pragma once

#include "types.hpp"
#include "utils.hpp"

#include <boost/pending/disjoint_sets.hpp>
#include <map>
#include <vector>
#include <iostream>

template<typename C, typename S, typename T>
inline bool try_merge(C & counts, S & sets, T s1, T s2, size_t size_threshold)
{
    using traits = watershed_traits<T>;
    auto real_size_s1 = counts[s1]&(~traits::on_border);
    auto real_size_s2 = counts[s2]&(~traits::on_border);
    if ( (real_size_s1 >= size_threshold) && (real_size_s2 >= size_threshold) ) {
        return false;
    }
    if (((traits::on_border&(counts[s1]|counts[s2]))==0)
          || (real_size_s1 >= size_threshold && counts[s2] < size_threshold)
          || (real_size_s2 >= size_threshold && counts[s1] < size_threshold)) {
    //if ((traits::on_border&(counts[s1]|counts[s2]))==0) {
        counts[s1] += counts[s2]&(~traits::on_border);
        counts[s1] |= counts[s2]&traits::on_border;
        counts[s2]  = 0;
        sets.link(s1, s2);
        T s = sets.find_set(s1);
        std::swap(counts[s], counts[s1]);
        return true;
    }
    else {
        counts[s1] |= counts[s2]&traits::on_border;
        counts[s2] |= counts[s1]&traits::on_border;
        return false;
    }
}

template< typename ID, typename F, typename L, typename M >
inline std::pair<std::vector<ID>, region_graph<ID,F>>
compute_merge( const region_graph<ID,F>& rg,
                            std::vector<std::size_t>& counts,
                            const L& tholds,
                            const M& lowt)
{
    using traits = watershed_traits<id_t>;
    memory_marker("merge: union begin");
    std::vector<ID> rank(counts.size());
    std::vector<ID> parent(counts.size());
    boost::disjoint_sets<ID*, ID*> sets(&rank[0], &parent[0]);
    for (ID i = 0; i < counts.size(); i++) {
        sets.make_set(i);
    }

    auto rit = rg.begin();

    std::size_t size = static_cast<std::size_t>(tholds.first);
    //F           thld = static_cast<F>(it.second);

    while ( (rit != rg.end()) && ( std::get<0>(*rit) > tholds.second) )
    {
        ID s1 = sets.find_set(std::get<1>(*rit));
        ID s2 = sets.find_set(std::get<2>(*rit));

        if ( s1 != s2 && s1 && s2 )
        {
            try_merge(counts, sets, s1, s2, size);
        }
        ++rit;
    }

    std::cout << "Done with merging" << std::endl;
    memory_marker("merge: union end / lut begin");

    std::vector<ID> remaps(counts.size());

    counts[0] &= ~traits::on_border;
    remaps[0] = 0;

    ID next_id = 1;

    std::size_t low = static_cast<std::size_t>(lowt);

    for ( ID id = 0; id < counts.size(); ++id )
    {
        ID s = sets.find_set(id);
        if ( counts[id]&(~traits::on_border) ) {
            if ( s && (counts[s] >= low) )
            {
                if (remaps[s] == 0) {
                    remaps[s] = next_id;
                    counts[next_id] = counts[s]&(~traits::on_border);
                    ++next_id;
                }
            } else {
                counts[s] = remaps[s] = 0;
            }
        }
    }

    counts.resize(next_id);

    std::vector<ID> lut(remaps.size());
    for (size_t id = 0; id < lut.size(); ++id)
        lut[id] = remaps[sets.find_set(id)];
    memory_marker("merge: lut end");
    free_container(rank);
    free_container(parent);
    free_container(remaps);

    std::cout << "Done with remapping, total: " << (next_id-1) << std::endl;

    region_graph<ID,F> new_rg;

    memory_marker("merge: new graph begin");

    std::vector<ID> rank_mst(next_id);
    std::vector<ID> parent_mst(next_id);
    boost::disjoint_sets<ID*, ID*> mst(&rank_mst[0], &parent_mst[0]);
    for (ID i = 0; i < next_id; i++) {
        mst.make_set(i);
    }

    for ( auto& it: rg )
    {
        ID s1 = lut[std::get<1>(it)];
        ID s2 = lut[std::get<2>(it)];
        ID a1 = mst.find_set(s1);
        ID a2 = mst.find_set(s2);

        if ( a1 != a2 && a1 && a2 && std::get<0>(it) > tholds.second)
        {
            mst.link(a1, a2);
            auto mm = std::minmax(s1,s2);
            new_rg.emplace_back(std::get<0>(it), mm.first, mm.second);
        }
    }

    memory_marker("merge: new graph end");
    std::cout << "Done with updating the region graph, size: "
              << new_rg.size() << std::endl;
    return {std::move(lut), std::move(new_rg)};
}
