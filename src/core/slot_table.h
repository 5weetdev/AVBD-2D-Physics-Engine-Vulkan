#pragma once

// A generational slot table: a vector of records with a free list. A record needs `bool alive`
// and `uint16_t gen`; releasing a slot bumps `gen` (never to 0), so a stale (index, generation)
// handle stops resolving. Handles are index1 = slot + 1 (0 is "no object").

#include <cstdint>
#include <vector>

template <class T> struct SlotTable
{
    std::vector<T> items;
    std::vector<int> freeList;
    int acquire()
    {
        int i;
        if (!freeList.empty())
        {
            i = freeList.back();
            freeList.pop_back();
        }
        else
        {
            items.emplace_back();
            i = (int)items.size() - 1;
        }
        items[(size_t)i].alive = true;
        return i;
    }
    T *find(int32_t index1, uint16_t gen)
    {
        if (index1 < 1 || index1 > (int32_t)items.size())
            return nullptr;
        T &t = items[(size_t)index1 - 1];
        return (t.alive && t.gen == gen) ? &t : nullptr;
    }
    void release(int i)
    {
        uint16_t g = (uint16_t)(items[(size_t)i].gen + 1);
        if (g == 0)
            g = 1;
        items[(size_t)i] = T();
        items[(size_t)i].gen = g;
        freeList.push_back(i);
    }
    int live() const { return (int)items.size() - (int)freeList.size(); }
};
