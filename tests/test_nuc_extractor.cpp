#include "../src/seg/NucExtractor.hpp"

#include <boost/multi_array.hpp>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

std::vector<nuc_wire_t> read_wires(const char * filename, size_t count)
{
    std::ifstream input(filename, std::ios_base::binary);
    std::vector<nuc_wire_t> wires(count);
    input.read(reinterpret_cast<char *>(wires.data()), count * sizeof(nuc_wire_t));
    assert(input.gcount() == static_cast<std::streamsize>(count * sizeof(nuc_wire_t)));
    assert(input.peek() == std::ifstream::traits_type::eof());
    return wires;
}

int main()
{
    boost::multi_array<nuc_t, 3> chunk(boost::extents[1][1][1]);
    const Coord coord{0, 0, 0};

    NucExtractor<seg_t, boost::multi_array<nuc_t, 3>> extractor(
        &chunk, nuc_ratio_t{3, 5}, 50);
    chunk[0][0][0] = 1;
    for (size_t i = 0; i < 5000; ++i) extractor.collectVoxel(coord, 100);
    chunk[0][0][0] = 2;
    for (size_t i = 0; i < 100; ++i) extractor.collectVoxel(coord, 100);
    chunk[0][0][0] = 1;
    for (size_t i = 0; i < 500; ++i) extractor.collectVoxel(coord, 200);
    chunk[0][0][0] = 2;
    for (size_t i = 0; i < 400; ++i) extractor.collectVoxel(coord, 200);
    chunk[0][0][0] = 3;
    for (size_t i = 0; i < 10; ++i) extractor.collectVoxel(coord, 300);

    const MapContainer<seg_t, seg_t> identity;
    const char * local_path = "nuc_extractor_test.data";
    extractor.output(identity, local_path);
    const auto wires = read_wires(local_path, 3);
    assert(wires[0].sid == 100 && wires[0].state == NUC_STATE_PROPER);
    assert(wires[0].id == 1 && wires[0].count == 5000 && wires[0].total == 5100);
    assert(wires[1].sid == 200 && wires[1].state == NUC_STATE_CONFLICT);
    assert(wires[1].count == 0 && wires[1].total == 900);
    assert(wires[2].sid == 300 && wires[2].state == NUC_STATE_NONE);
    assert(wires[2].count == 0 && wires[2].total == 0);

    NucExtractor<seg_t, boost::multi_array<nuc_t, 3>> remapped(
        &chunk, nuc_ratio_t{3, 5}, 1);
    chunk[0][0][0] = 7;
    for (size_t i = 0; i < 60; ++i) remapped.collectVoxel(coord, 10);
    chunk[0][0][0] = 8;
    for (size_t i = 0; i < 60; ++i) remapped.collectVoxel(coord, 20);
    MapContainer<seg_t, seg_t> chunk_map;
    chunk_map[10] = 30;
    chunk_map[20] = 30;
    const char * remapped_path = "nuc_extractor_remapped_test.data";
    remapped.output(chunk_map, remapped_path);
    const auto joined = read_wires(remapped_path, 1);
    assert(joined[0].sid == 30 && joined[0].state == NUC_STATE_CONFLICT);
    assert(joined[0].id == 0 && joined[0].count == 0 && joined[0].total == 120);

    std::filesystem::remove(local_path);
    std::filesystem::remove(remapped_path);
    std::cout << "test_nuc_extractor: PASS" << std::endl;
}
