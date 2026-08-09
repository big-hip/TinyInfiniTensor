#include "core/allocator.h"
#include <utility>

namespace infini
{
    Allocator::Allocator(Runtime runtime) : runtime(runtime)
    {
        used = 0;
        peak = 0;
        ptr = nullptr;

        // 'alignment' defaults to sizeof(uint64_t), because it is the length of
        // the longest data type currently supported by the DataType field of
        // the tensor
        alignment = sizeof(uint64_t);
    }

    Allocator::~Allocator()
    {
        if (this->ptr != nullptr)
        {
            runtime->dealloc(this->ptr);
        }
    }

    size_t Allocator::alloc(size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        // pad the size to the multiple of alignment
        size = this->getAlignedSize(size);

        if (size == 0)
            return peak;

        // First-fit allocation keeps the simulation deterministic while the
        // address-ordered map makes splitting and coalescing straightforward.
        for (auto it = freeBlocks.begin(); it != freeBlocks.end(); ++it)
        {
            if (it->second < size)
                continue;

            const auto addr = it->first;
            const auto blockSize = it->second;
            freeBlocks.erase(it);
            if (blockSize > size)
                freeBlocks.emplace(addr + size, blockSize - size);
            used += size;
            return addr;
        }

        const auto addr = peak;
        peak += size;
        used += size;
        return addr;
    }

    void Allocator::free(size_t addr, size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);

        if (size == 0)
            return;
        IT_ASSERT(addr <= peak && size <= peak - addr);
        IT_ASSERT(size <= used);

        used -= size;
        auto next = freeBlocks.lower_bound(addr);
        IT_ASSERT(next == freeBlocks.end() || addr + size <= next->first);

        if (next != freeBlocks.begin())
        {
            auto prev = std::prev(next);
            if (prev->first + prev->second == addr)
            {
                addr = prev->first;
                size += prev->second;
                freeBlocks.erase(prev);
            }
        }

        if (next != freeBlocks.end() && addr + size == next->first)
        {
            size += next->second;
            freeBlocks.erase(next);
        }

        // A free block at the end does not contribute to the arena's high
        // water mark. Rewind it so a larger subsequent request can grow from
        // the same address instead of unnecessarily extending the arena.
        if (addr + size == peak)
        {
            peak = addr;
            return;
        }
        freeBlocks.emplace(addr, size);
    }

    void *Allocator::getPtr()
    {
        if (this->ptr == nullptr)
        {
            this->ptr = runtime->alloc(this->peak);
            printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
        }
        return this->ptr;
    }

    size_t Allocator::getAlignedSize(size_t size)
    {
        if (size == 0)
            return 0;
        return ((size - 1) / this->alignment + 1) * this->alignment;
    }

    void Allocator::info()
    {
        std::cout << "Used memory: " << this->used
                  << ", peak memory: " << this->peak << std::endl;
    }
}
