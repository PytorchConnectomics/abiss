#ifndef NUC_EXTRACTOR_HPP
#define NUC_EXTRACTOR_HPP

#include "Types.h"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

template<typename Tseg, typename Chunk>
class NucExtractor
{
public:
    NucExtractor(const Chunk * nuc, nuc_ratio_t ratio, uint64_t min_tagged)
        : m_nuc(nuc), m_ratio(ratio), m_min_tagged(min_tagged) {}

    void collectVoxel(Coord c, Tseg segid)
    {
        if (m_nuc == nullptr) {
            return;
        }
        const nuc_t id = (*m_nuc)[c[0]][c[1]][c[2]];
        if (id != 0) {
            auto & count = m_counts[segid][id];
            count = nuc_add(count, 1);
        }
    }

    void collectBoundary(int face, Coord c, Tseg segid) {}
    void collectContactingSurface(int nv, Coord c, Tseg segid1, Tseg segid2) {}

    void output(const MapContainer<Tseg, Tseg> & chunk_map, const std::string & filename)
    {
        MapContainer<Tseg, MapContainer<nuc_t, uint64_t>> remapped_counts;
        for (const auto & [sid, counts] : m_counts) {
            const Tseg target = chunk_map.contains(sid) ? chunk_map.at(sid) : sid;
            for (const auto & [id, count] : counts) {
                auto & target_count = remapped_counts[target][id];
                target_count = nuc_add(target_count, count);
            }
        }

        uint64_t conflict_sv = 0;
        uint64_t minority_sv = 0;
        uint64_t subfloor_sv = 0;
        uint64_t subfloor_voxels = 0;
        std::vector<nuc_wire_t> output;
        output.reserve(remapped_counts.size());

        for (const auto & [sid, counts] : remapped_counts) {
            uint64_t tagged = 0;
            uint64_t max_count = 0;
            nuc_t max_id = 0;
            for (const auto & [id, count] : counts) {
                tagged = nuc_add(tagged, count);
                if (count > max_count || (count == max_count && id < max_id)) {
                    max_count = count;
                    max_id = id;
                }
            }

            nuc_record_t record;
            if (tagged < m_min_tagged) {
                subfloor_sv = nuc_add(subfloor_sv, 1);
                subfloor_voxels = nuc_add(subfloor_voxels, tagged);
            } else if (nuc_is_dominant(max_count, tagged, m_ratio)) {
                record = nuc_record_t{NUC_STATE_PROPER, max_id, max_count, tagged};
                if (tagged > max_count) {
                    minority_sv = nuc_add(minority_sv, 1);
                }
            } else {
                record = nuc_record_t{NUC_STATE_CONFLICT, 0, 0, tagged};
                conflict_sv = nuc_add(conflict_sv, 1);
            }
            output.push_back(make_nuc_wire(sid, record));
        }

        std::sort(output.begin(), output.end(),
                  [](const auto & a, const auto & b) { return a.sid < b.sid; });
        std::ofstream ofs(filename, std::ios_base::binary);
        assert(ofs.is_open());
        for (const auto & wire : output) {
            ofs.write(reinterpret_cast<const char *>(&wire), sizeof(wire));
        }
        assert(!ofs.bad());
        ofs.close();

        if (m_nuc != nullptr) {
            std::cout << "nuc: conflict_sv " << conflict_sv << std::endl;
            std::cout << "nuc: minority_sv " << minority_sv << std::endl;
            std::cout << "nuc: subfloor_sv " << subfloor_sv << std::endl;
            std::cout << "nuc: subfloor_voxels " << subfloor_voxels << std::endl;
        }
    }

private:
    // Nucleus masks are expected to be sparse, so the per-supervoxel inner maps
    // only exist for supervoxels that contain tagged voxels.
    const Chunk * m_nuc;
    nuc_ratio_t m_ratio;
    uint64_t m_min_tagged;
    MapContainer<Tseg, MapContainer<nuc_t, uint64_t>> m_counts;
};

#endif
