#pragma once

// A small persistent thread pool for the host-side loops the 2D world runs over the
// moved-body list (VkWorld2D::refreshHostCopies) and over the C API's event/view pass.
// Those loops are index range work: split [0, count) into consecutive chunks and let each
// worker take one. Work items are numbered, and a caller that needs a deterministic result
// appends per-chunk output in chunk order after the pass, which is why the chunk index is
// handed back to the callback.
//
// The pool exists so a step does not create threads: it is built once with the world and
// reused. `run` blocks until every chunk is done, so callers stay simple.

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace avbd2d
{
class TaskPool2D
{
public:
    // threads == 0 picks one worker per hardware thread, capped at 8 (the loops here are
    // memory-bound, and more threads than that do not help). At least one worker exists.
    explicit TaskPool2D(size_t threads = 0)
    {
        const size_t hw = threads ? threads : std::max(1u, std::thread::hardware_concurrency());
        const size_t workers = std::min<size_t>(std::max<size_t>(hw, 1), 8) - 1;
        m_workers.reserve(workers);
        for (size_t i = 0; i < workers; i++)
            m_workers.emplace_back([this] { workerLoop(); });
    }

    ~TaskPool2D()
    {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_stop = true;
        }
        m_cvWork.notify_all();
        for (std::thread &t : m_workers)
            if (t.joinable())
                t.join();
    }

    TaskPool2D(const TaskPool2D &) = delete;
    TaskPool2D &operator=(const TaskPool2D &) = delete;

    // Workers plus the calling thread.
    size_t threadCount() const { return m_workers.size() + 1; }

    // The chunks [0, count) splits into, of which the last may be shorter. The C API sizes
    // its per-chunk output vectors with this, so it must agree with run().
    static size_t chunkCount(size_t count, size_t chunk)
    {
        if (count == 0)
            return 0;
        return chunk == 0 ? 1 : (count + chunk - 1) / chunk;
    }

    // Calls fn(chunk, begin, end) for every chunk of [0, count) and returns once all of them
    // have finished. 0 means "no chunk size": one chunk. The callback may capture anything
    // by reference; it must be safe to call concurrently and must not throw.
    template <typename Fn> void run(size_t count, size_t chunk, Fn &&fn)
    {
        if (count == 0)
            return;
        chunk = chunk ? chunk : count;
        const size_t chunks = chunkCount(count, chunk);
        if (chunks == 1 || m_workers.empty())
        {
            for (size_t c = 0; c < chunks; c++)
            {
                const size_t begin = c * chunk;
                fn(c, begin, std::min(count, begin + chunk));
            }
            return;
        }
        const std::function<void(size_t)> body = [&fn, count, chunk](size_t c)
        {
            const size_t begin = c * chunk;
            fn(c, begin, std::min(count, begin + chunk));
        };
        runChunks(chunks, body);
    }

private:
    void runChunks(size_t chunks, const std::function<void(size_t)> &body)
    {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_body = &body;
            m_chunkTotal.store(chunks, std::memory_order_relaxed);
            m_next.store(0, std::memory_order_relaxed);
            m_pending = m_workers.size() + 1; // the calling thread is one of the workers
            ++m_generation;
        }
        m_cvWork.notify_all();
        drain();
        std::unique_lock<std::mutex> lock(m_mutex);
        if (--m_pending > 0)
            m_cvDone.wait(lock, [this] { return m_pending == 0; });
        m_body = nullptr;
    }

    void drain()
    {
        const size_t total = m_chunkTotal.load(std::memory_order_relaxed);
        const std::function<void(size_t)> *body = m_body;
        for (;;)
        {
            const size_t c = m_next.fetch_add(1, std::memory_order_relaxed);
            if (c >= total)
                break;
            (*body)(c);
        }
    }

    void workerLoop()
    {
        size_t seen = 0;
        for (;;)
        {
            {
                std::unique_lock<std::mutex> lock(m_mutex);
                m_cvWork.wait(lock, [&] { return m_stop || m_generation != seen; });
                if (m_stop)
                    return;
                seen = m_generation;
            }
            drain();
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                if (--m_pending == 0)
                    m_cvDone.notify_all();
            }
        }
    }

    std::vector<std::thread> m_workers;
    std::mutex m_mutex;
    std::condition_variable m_cvWork;
    std::condition_variable m_cvDone;
    const std::function<void(size_t)> *m_body = nullptr;
    std::atomic<size_t> m_chunkTotal{0};
    std::atomic<size_t> m_next{0};
    size_t m_pending = 0;
    size_t m_generation = 0;
    bool m_stop = false;
};
} // namespace avbd2d
