#ifndef TYPES_H
#define TYPES_H

#include <array>
#include <boost/multi_array.hpp>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>

#include "../global_types.h"

typedef boost::multi_array_types::extent_range Range;

template <typename T, int n>
using ConstChunkRef = boost::const_multi_array_ref<T, n, const T*>;

template <class Ts>
using SegPair = std::pair<Ts, Ts>;

using Coord = std::array<int64_t, 3>;

using ContactRegion = SetContainer<Coord, HashFunction<Coord> >;
using ContactRegionExt = MapContainer<Coord, int, HashFunction<Coord> >;

template <class Ta>
using Edge = std::array<MapContainer<Coord, Ta, HashFunction<Coord> >, 3>;

using semantic_t = uint8_t;
using nuc_t = uint32_t;

enum : uint8_t {
    NUC_STATE_NONE = 0,
    NUC_STATE_PROPER = 1,
    NUC_STATE_CONFLICT = 2,
};

// Reachable records preserve:
//   PROPER => count * ratio.den >= ratio.num * total
//   NONE   => count == 0 && total == 0
//   CONFLICT => count == 0
struct __attribute__((packed)) nuc_record_t
{
    uint8_t state = NUC_STATE_NONE;
    uint32_t id = 0;
    uint64_t count = 0;
    uint64_t total = 0;
};

struct __attribute__((packed)) nuc_wire_t
{
    seg_t sid;
    uint8_t state;
    uint32_t id;
    uint64_t count;
    uint64_t total;
};

static_assert(sizeof(nuc_wire_t) == 29);
static_assert(offsetof(nuc_wire_t, sid) == 0);
static_assert(offsetof(nuc_wire_t, state) == 8);
static_assert(offsetof(nuc_wire_t, id) == 9);
static_assert(offsetof(nuc_wire_t, count) == 13);
static_assert(offsetof(nuc_wire_t, total) == 21);

struct nuc_ratio_t
{
    uint64_t num = 3;
    uint64_t den = 5;
};

inline bool nuc_is_dominant(uint64_t count, uint64_t total, const nuc_ratio_t & ratio)
{
    return static_cast<__uint128_t>(count) * ratio.den
        >= static_cast<__uint128_t>(ratio.num) * total;
}

inline uint64_t nuc_add(uint64_t a, uint64_t b)
{
    if (a > UINT64_MAX - b) {
        std::cerr << "nuc: voxel count overflow: " << a << " + " << b << std::endl;
        std::abort();
    }
    return a + b;
}

inline nuc_record_t nuc_join(const nuc_record_t & a, const nuc_record_t & b)
{
    nuc_record_t result;
    result.total = nuc_add(a.total, b.total);
    if (a.state == NUC_STATE_CONFLICT || b.state == NUC_STATE_CONFLICT) {
        result.state = NUC_STATE_CONFLICT;
    } else if (a.state == NUC_STATE_NONE) {
        result.state = b.state;
        result.id = b.id;
        result.count = b.count;
    } else if (b.state == NUC_STATE_NONE) {
        result.state = a.state;
        result.id = a.id;
        result.count = a.count;
    } else if (a.id == b.id) {
        result.state = NUC_STATE_PROPER;
        result.id = a.id;
        result.count = nuc_add(a.count, b.count);
    } else {
        result.state = NUC_STATE_CONFLICT;
    }
    return result;
}

inline bool nuc_can_merge(const nuc_record_t & a, const nuc_record_t & b)
{
    const bool conflict_a = a.state == NUC_STATE_CONFLICT;
    const bool conflict_b = b.state == NUC_STATE_CONFLICT;
    if (conflict_a && conflict_b) {
        return false; // Invariant D, clause 3.
    }
    if (conflict_a) {
        return b.state == NUC_STATE_NONE; // Invariant D, clause 2.
    }
    if (conflict_b) {
        return a.state == NUC_STATE_NONE; // Invariant D, clause 2.
    }
    if (a.state == NUC_STATE_NONE || b.state == NUC_STATE_NONE) {
        return true;
    }
    return a.id == b.id; // Invariant D, clause 1.
}

inline nuc_record_t nuc_record_from_wire(const nuc_wire_t & wire)
{
    return nuc_record_t{wire.state, wire.id, wire.count, wire.total};
}

inline nuc_wire_t make_nuc_wire(seg_t sid, const nuc_record_t & record)
{
    return nuc_wire_t{sid, record.state, record.id, record.count, record.total};
}

inline bool parse_nuc_ratio(const char * value, nuc_ratio_t & ratio)
{
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    errno = 0;
    char * end = nullptr;
    const double parsed = std::strtod(value, &end);
    if (errno == ERANGE || end == value || *end != '\0' || !std::isfinite(parsed)
        || parsed <= 0.5 || parsed > 1.0) {
        return false;
    }
    ratio.num = static_cast<uint64_t>(std::llround(parsed * 1000.0));
    ratio.den = 1000;
    return true;
}

inline nuc_ratio_t nuc_ratio_from_env()
{
    nuc_ratio_t ratio;
    const char * value = std::getenv("ABISS_NUC_DOMINANCE");
    if (value != nullptr && !parse_nuc_ratio(value, ratio)) {
        std::cerr << "nuc: ABISS_NUC_DOMINANCE must be finite and in (0.5, 1.0], got "
                  << value << std::endl;
        std::abort();
    }
    return ratio;
}

inline uint64_t nuc_min_tagged_from_env()
{
    const char * value = std::getenv("ABISS_NUC_MIN_TAGGED");
    if (value == nullptr) {
        return 50;
    }
    if (value[0] == '\0' || value[0] == '-') {
        std::cerr << "nuc: ABISS_NUC_MIN_TAGGED must be an unsigned integer, got "
                  << value << std::endl;
        std::abort();
    }
    errno = 0;
    char * end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (errno == ERANGE || end == value || *end != '\0') {
        std::cerr << "nuc: ABISS_NUC_MIN_TAGGED must be an unsigned integer, got "
                  << value << std::endl;
        std::abort();
    }
    return static_cast<uint64_t>(parsed);
}

template <class T>
struct __attribute__((packed)) matching_entry_t
{
    T oid;
    size_t boundary_size;
    T nid;
    size_t agg_size;
};

template <class T_seg, class T_aff>
struct __attribute__((packed)) rg_entry
{
    T_seg s1;
    T_seg s2;
    T_aff aff;
    size_t area;
    rg_entry() = default;
    explicit rg_entry(const std::pair<SegPair<T_seg>, std::pair<T_aff, size_t> >  & p) {
        s1 = p.first.first;
        s2 = p.first.second;
        aff = p.second.first;
        area = p.second.second;
    }
};

#endif
