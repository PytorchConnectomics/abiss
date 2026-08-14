#include "../src/seg/Types.h"

#include <cassert>
#include <csignal>
#include <cstdint>
#include <iostream>
#include <limits>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

bool equal_record(const nuc_record_t & a, const nuc_record_t & b)
{
    return a.state == b.state && a.id == b.id && a.count == b.count
        && a.total == b.total;
}

void assert_closed(const nuc_record_t & record, const nuc_ratio_t & ratio)
{
    if (record.state == NUC_STATE_PROPER) {
        assert(nuc_is_dominant(record.count, record.total, ratio));
    } else {
        assert(record.count == 0);
        if (record.state == NUC_STATE_NONE) {
            assert(record.total == 0);
        }
    }
}

int main()
{
    const nuc_ratio_t ratio{3, 5};
    const nuc_record_t none{};
    const nuc_record_t proper1a{NUC_STATE_PROPER, 1, 3, 5};
    const nuc_record_t proper1b{NUC_STATE_PROPER, 1, 9, 10};
    const nuc_record_t proper2{NUC_STATE_PROPER, 2, 4, 5};
    const nuc_record_t conflict{NUC_STATE_CONFLICT, 0, 0, 9};
    const std::vector<nuc_record_t> records{
        none, proper1a, proper1b, proper2, conflict};

    for (const auto & a : records) {
        assert_closed(a, ratio);
        const auto doubled = nuc_join(a, a);
        assert(doubled.state == a.state);
        assert(doubled.id == a.id);
        assert(doubled.count == 2 * a.count);
        assert(doubled.total == 2 * a.total);
        assert_closed(doubled, ratio);
        for (const auto & b : records) {
            const auto ab = nuc_join(a, b);
            const auto ba = nuc_join(b, a);
            assert(equal_record(ab, ba));
            assert_closed(ab, ratio);
            for (const auto & c : records) {
                const auto left = nuc_join(nuc_join(a, b), c);
                const auto right = nuc_join(a, nuc_join(b, c));
                assert(equal_record(left, right));
                assert_closed(left, ratio);
            }
        }
    }

    const uint64_t two_to_53 = uint64_t{1} << 53;
    assert(!nuc_is_dominant(two_to_53, two_to_53 + 1, nuc_ratio_t{1, 1}));

    const pid_t child = fork();
    assert(child >= 0);
    if (child == 0) {
        nuc_add(std::numeric_limits<uint64_t>::max(), 1);
        _exit(0);
    }
    int status = 0;
    assert(waitpid(child, &status, 0) == child);
    assert(WIFSIGNALED(status));
    assert(WTERMSIG(status) == SIGABRT);

    nuc_ratio_t parsed;
    assert(parse_nuc_ratio("0.6", parsed));
    assert(parsed.num == 600 && parsed.den == 1000);
    assert(!parse_nuc_ratio("NaN", parsed));
    assert(!parse_nuc_ratio("inf", parsed));
    assert(!parse_nuc_ratio("0.5", parsed));
    assert(!parse_nuc_ratio("1.1", parsed));

    const nuc_record_t p1{NUC_STATE_PROPER, 1, 50, 50};
    const nuc_record_t p1_again{NUC_STATE_PROPER, 1, 60, 60};
    const nuc_record_t p2{NUC_STATE_PROPER, 2, 50, 50};
    assert(nuc_can_merge(none, none));
    assert(nuc_can_merge(none, p1));
    assert(nuc_can_merge(none, conflict));
    assert(nuc_can_merge(p1, p1_again));
    assert(!nuc_can_merge(p1, p2));
    assert(!nuc_can_merge(conflict, p1));
    assert(!nuc_can_merge(p1, conflict));
    assert(!nuc_can_merge(conflict, conflict));

    std::cout << "test_nuc_algebra: PASS" << std::endl;
}
