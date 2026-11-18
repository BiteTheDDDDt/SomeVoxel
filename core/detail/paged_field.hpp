#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <vector>

namespace somevoxel {

// Zero-filled pages for fields that only touch a subset of the lattice.
template <typename Value, size_t Length, bool ReclaimEmpty = false>
class PagedField {
    static constexpr size_t page_size = 4096 / sizeof(Value);
    static constexpr size_t page_count = (Length + page_size - 1) / page_size;

    struct Page {
        std::array<Value, page_size> values{};
        size_t nonzero = 0;
    };

    std::vector<std::unique_ptr<Page>> pages;
    size_t allocated = 0;

public:
    bool empty() const { return allocated == 0; }

    size_t bytes() const { return pages.capacity() * sizeof(std::unique_ptr<Page>) + allocated * sizeof(Page); }

    void clear() {
        pages.clear();
        allocated = 0;
    }

    Value get(size_t index) const {
        if (pages.empty()) return {};
        const auto &page = pages[index / page_size];
        return page ? page->values[index % page_size] : Value{};
    }

    void set(size_t index, Value value) {
        if (pages.empty()) {
            if (value == Value{}) return;
            pages.resize(page_count);
        }
        auto &page = pages[index / page_size];
        if (!page) {
            if (value == Value{}) return;
            page = std::make_unique<Page>();
            ++allocated;
        }
        Value &slot = page->values[index % page_size];
        if constexpr (ReclaimEmpty) {
            if (slot == Value{} && value != Value{}) ++page->nonzero;
            if (slot != Value{} && value == Value{} && --page->nonzero == 0) {
                page.reset();
                --allocated;
                return;
            }
        }
        slot = value;
    }
};

} // namespace somevoxel
