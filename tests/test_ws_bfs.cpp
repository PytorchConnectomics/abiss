#include "../src/ws/basic_watershed.hpp"
#include "../src/ws/region_graph.hpp"
#include <cstring>
#include <random>
#include <stdexcept>

void check(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

template<typename ID>
void queues() {
    std::mt19937 rng(21);
    std::vector<float> data(22*26*18*3);
    auto aff = std::make_shared<affinity_graph<float>>(
        data.data(), boost::extents[22][26][18][3], boost::fortran_storage_order());
    for (int fixture = 0; fixture < 4; ++fixture) {
        for (auto& a : data) a = fixture == 0 ? 0 : float(rng()%11)/10;
        if (fixture == 2) std::fill(data.begin(), data.end(), 0.6f);
        std::array<bool,6> flags;
        for (size_t i = 0; i < flags.size(); ++i) flags[i] = fixture == 3 ? i%2 : fixture != 2;
        auto narrow = watershed<ID,uint32_t>(aff, 0.05f, 0.95f, flags);
        auto wide = watershed<ID,std::ptrdiff_t>(aff, 0.05f, 0.95f, flags);
        check(std::get<1>(narrow) == std::get<1>(wide), "queue counts differ");
        check(std::memcmp(std::get<0>(narrow)->data(), std::get<0>(wide)->data(),
                          22*26*18*sizeof(ID)) == 0, "queue labels differ");
    }
}

template<EdgeScoreMode Mode>
void scores() {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    for (auto values : std::vector<std::vector<float>>{
             {0.1f,0.2f,0.3f,1e-7f,0.1f,1e-7f}, {1e20f,1,-1e20f,0.3f},
             {-0.0f,0.0f}, {0.0f,-0.0f}, {nan,1,2}, {1,nan,2}, {2,1,nan}}) {
        EdgeAccumulator<float,Mode> acc;
        for (float a : values) acc.push_back(a);
        EdgeScoreConfig cfg;
        cfg.mode = Mode;
        float actual = acc.score(cfg), expected = compute_edge_score(values,cfg);
        check(std::memcmp(&actual, &expected, sizeof(float)) == 0, "score field bytes differ");
    }
}

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--dend-layout") {
        region_graph<seg_t,aff_t>::value_type t;
        const auto base = reinterpret_cast<const char*>(&t);
        std::cout << "{\"size\":" << sizeof(t)
                  << ",\"score\":" << reinterpret_cast<const char*>(&std::get<0>(t))-base
                  << ",\"id1\":" << reinterpret_cast<const char*>(&std::get<1>(t))-base
                  << ",\"id2\":" << reinterpret_cast<const char*>(&std::get<2>(t))-base
                  << "}" << std::endl;
        return 0;
    }
    try {
        const size_t limit = size_t(1)<<32;
        check(bfs_fits_u32(limit-1) && bfs_fits_u32(limit) && !bfs_fits_u32(limit+1), "queue cutoff");
        size_t p;
        check(watershed_size_valid<uint32_t>((limit>>1)-1,1,1,p), "ws high_bit-1");
        check(!watershed_size_valid<uint32_t>(limit>>1,1,1,p), "ws high_bit");
        check(!watershed_size_valid<uint32_t>((limit>>1)+1,1,1,p), "ws high_bit+1");
        const size_t high64 = size_t(1)<<63;
        check(watershed_size_valid<uint64_t>(high64-1,1,1,p), "ws64 high_bit-1");
        check(!watershed_size_valid<uint64_t>(high64,1,1,p), "ws64 high_bit");
        check(!watershed_size_valid<uint64_t>(high64+1,1,1,p), "ws64 high_bit+1");
        check(!watershed_size_valid<uint64_t>(SIZE_MAX,2,2,p), "product overflow");
        queues<uint32_t>();
        queues<uint64_t>();
        scores<EdgeScoreMode::MAX>();
        scores<EdgeScoreMode::MEAN>();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
