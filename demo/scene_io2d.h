#pragma once

// Scene capture for the 2D solver: freezes a running world into a Scene2D and stores it in a file,
// so a state found in the demo (a jam, a explosion, a slow pile) replays as an ordinary scene.
// Host-only: knows Scene2D and nothing of Vulkan.
//
//   captureScene2D   the authored scene + the live poses / velocities -> a self-contained Scene2D.
//                    Bodies the world removed and joints that broke are dropped and the indices
//                    compacted; no-collide pairs between the survivors stay (a broken joint's pair
//                    keeps ignoring each other, as it does in the live world).
//   saveScene2D      Scene2D + solver parameters -> a binary file.
//   loadScene2D      the inverse; the file is validated, a bad one is refused with a reason.
//
// What a capture does not hold: sleep state (every body starts awake and re-sleeps by the normal
// rule), warm-start multipliers and the contact cache (they rebuild in a few steps), motor/limit
// edits queued on a live joint, and a grab in progress.
//
// The file is the raw arrays of Scene2D, each prefixed by its count and element size, so a build
// whose Scene2D, Vec2 or JointAxis2 layout differs refuses an old file ("element size") instead of
// misreading it. Every Scene2D array is listed once, in visitScene2D(): a new array added to
// Scene2D must be added there too, or a capture silently loses it.

#include "scene2d.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace avbd2d
{

struct CaptureMeta2D
{
    std::string source; // the scene the capture came from
    uint64_t step = 0;  // simulation steps taken when it was captured
};

constexpr char kCaptureExt2D[] = ".a2dcap";

// Calls f(array) for every array of the scene, in the file's order.
template <class S, class F>
void visitScene2D(S &s, F &&f)
{
    f(s.pos); f(s.vel); f(s.half); f(s.localCenter);
    f(s.ang); f(s.angVel); f(s.mass); f(s.moment);
    f(s.bodyType); f(s.lockX); f(s.lockY); f(s.lockRot);
    f(s.gravityScale); f(s.linearDamping); f(s.angularDamping);
    f(s.color); f(s.shapeFirst); f(s.shapeCount); f(s.isBullet);
    f(s.shType); f(s.shOff); f(s.shHalf);
    f(s.shAng); f(s.shRad); f(s.shFric);
    f(s.shVOff); f(s.shVCnt); f(s.shBody);
    f(s.shCat); f(s.shMask); f(s.shGroup); f(s.shUserId);
    f(s.shTangentSpeed); f(s.shRolling); f(s.shRestitution);
    f(s.verts); f(s.norms); f(s.shFlags);
    f(s.jointA); f(s.jointB); f(s.jointRA); f(s.jointRB);
    f(s.jointFrameA); f(s.jointRest); f(s.jointArm); f(s.jointFrac); f(s.jointAxis);
    f(s.ignore);
}

// The authored scene `src` with the live state applied. pos / ang are indexed by body (as
// VkWorld2D::positions() / angles()), vel / angVel likewise (velocities() / angularVelocities());
// removed / broken are VkWorld2D::downloadLiveFlags(). Any of the four state arrays may be null
// (the authored value is kept). Static bodies keep their authored pose; a kinematic one takes the
// live pose and keeps its authored velocity.
inline Scene2D captureScene2D(const Scene2D &src, const Vec2 *pos, const float *ang, const Vec2 *vel,
                              const float *angVel, const std::vector<uint8_t> &removed,
                              const std::vector<uint8_t> &broken)
{
    Scene2D out;
    out.params = src.params;
    out.cameraCenter = src.cameraCenter;
    out.cameraHeight = src.cameraHeight;

    const int n = src.bodyCount();
    std::vector<int> remap((size_t)n, -1);
    for (int i = 0; i < n; i++)
    {
        const size_t b = (size_t)i;
        if (b < removed.size() && removed[b])
            continue;
        const int id = out.appendBody(src, i, Vec2());
        remap[b] = id;
        const size_t d = (size_t)id;
        if (src.bodyType[b] == BODY2D_STATIC)
            continue;
        if (pos)
            out.pos[d] = pos[b];
        if (ang)
            out.ang[d] = ang[b];
        if (src.bodyType[b] == BODY2D_DYNAMIC)
        {
            if (vel)
                out.vel[d] = vel[b];
            if (angVel)
                out.angVel[d] = angVel[b];
        }
    }

    for (int j = 0; j < src.jointCount(); j++)
    {
        const size_t q = (size_t)j;
        if (q < broken.size() && broken[q])
            continue;
        const int a = src.jointA[q], b = src.jointB[q];
        if ((a >= 0 && remap[(size_t)a] < 0) || (b >= 0 && remap[(size_t)b] < 0))
            continue;
        out.jointA.push_back(a >= 0 ? remap[(size_t)a] : -1);
        out.jointB.push_back(b >= 0 ? remap[(size_t)b] : -1);
        out.jointRA.push_back(src.jointRA[q]);
        out.jointRB.push_back(src.jointRB[q]);
        out.jointFrameA.push_back(src.jointFrameA[q]);
        out.jointRest.push_back(src.jointRest[q]);
        out.jointArm.push_back(src.jointArm[q]);
        out.jointFrac.push_back(src.jointFrac[q]);
        for (size_t k = 0; k < 3; k++)
            out.jointAxis.push_back(src.jointAxis[3 * q + k]);
    }

    for (const uint64_t key : src.ignore)
    {
        const int a = remap[(size_t)(key >> 32)], b = remap[(size_t)(key & 0xffffffffu)];
        if (a >= 0 && b >= 0)
            out.ignoreCollision(a, b);
    }
    out.finalize();
    return out;
}

namespace scene_io_detail
{

constexpr char kMagic[8] = {'A', '2', 'D', 'C', 'A', 'P', 0, 1};

struct FileCloser
{
    void operator()(FILE *f) const { if (f) fclose(f); }
};
using File = std::unique_ptr<FILE, FileCloser>;

struct Writer
{
    FILE *f;
    bool ok = true;
    void raw(const void *p, size_t bytes)
    {
        if (ok && bytes > 0)
            ok = fwrite(p, 1, bytes, f) == bytes;
    }
    template <class T>
    void pod(const T &v)
    {
        static_assert(std::is_trivially_copyable<T>::value, "");
        raw(&v, sizeof(T));
    }
    template <class T>
    void array(const std::vector<T> &v)
    {
        static_assert(std::is_trivially_copyable<T>::value, "");
        pod<uint64_t>(v.size());
        pod<uint32_t>(sizeof(T));
        raw(v.data(), v.size() * sizeof(T));
    }
    void str(const std::string &s)
    {
        pod<uint32_t>((uint32_t)s.size());
        raw(s.data(), s.size());
    }
};

struct Reader
{
    FILE *f;
    uint64_t left; // bytes still in the file
    std::string err;
    bool raw(void *p, size_t bytes)
    {
        if (!err.empty())
            return false;
        if (bytes > left || (bytes > 0 && fread(p, 1, bytes, f) != bytes))
        {
            err = "file is truncated";
            return false;
        }
        left -= bytes;
        return true;
    }
    template <class T>
    bool pod(T &v)
    {
        return raw(&v, sizeof(T));
    }
    template <class T>
    bool array(std::vector<T> &v)
    {
        uint64_t count = 0;
        uint32_t elem = 0;
        if (!pod(count) || !pod(elem))
            return false;
        if (elem != sizeof(T))
        {
            err = "element size differs (captured by a different build)";
            return false;
        }
        if (count > left / sizeof(T))
        {
            err = "file is truncated";
            return false;
        }
        v.resize((size_t)count);
        return raw(v.data(), (size_t)count * sizeof(T));
    }
    bool str(std::string &s)
    {
        uint32_t n = 0;
        if (!pod(n) || n > left)
        {
            if (err.empty())
                err = "file is truncated";
            return false;
        }
        s.resize(n);
        return raw(s.data(), n);
    }
};

// Cross-checks the arrays of a loaded scene so a corrupt file cannot index out of range.
inline bool validate(const Scene2D &s, std::string &err)
{
    const size_t N = s.pos.size(), M = s.shBody.size(), J = s.jointA.size();
    bool ok = true;
    const size_t bodyArrays[] = {s.vel.size(), s.half.size(), s.localCenter.size(), s.ang.size(), s.angVel.size(),
                                 s.mass.size(), s.moment.size(), s.bodyType.size(), s.lockX.size(), s.lockY.size(),
                                 s.lockRot.size(), s.gravityScale.size(), s.linearDamping.size(),
                                 s.angularDamping.size(), s.color.size(), s.shapeFirst.size(), s.shapeCount.size(),
                                 s.isBullet.size()};
    for (size_t c : bodyArrays)
        ok = ok && c == N;
    const size_t shapeArrays[] = {s.shType.size(), s.shOff.size(), s.shHalf.size(), s.shAng.size(), s.shRad.size(),
                                  s.shFric.size(), s.shVOff.size(), s.shVCnt.size(), s.shCat.size(), s.shMask.size(),
                                  s.shGroup.size(), s.shUserId.size(), s.shTangentSpeed.size(), s.shRolling.size(),
                                  s.shRestitution.size(), s.shFlags.size()};
    for (size_t c : shapeArrays)
        ok = ok && c == M;
    const size_t jointArrays[] = {s.jointB.size(), s.jointRA.size(), s.jointRB.size(), s.jointFrameA.size(),
                                  s.jointRest.size(), s.jointArm.size(), s.jointFrac.size()};
    for (size_t c : jointArrays)
        ok = ok && c == J;
    ok = ok && s.jointAxis.size() == 3 * J && s.verts.size() == s.norms.size();
    if (!ok)
    {
        err = "array sizes disagree";
        return false;
    }
    for (size_t i = 0; i < N; i++)
        if (s.shapeCount[i] < 0 || s.shapeFirst[i] < 0 || (size_t)s.shapeFirst[i] + (size_t)s.shapeCount[i] > M)
        {
            err = "a body's shape range is out of bounds";
            return false;
        }
    for (size_t k = 0; k < M; k++)
        if (s.shBody[k] >= (int)N || s.shVCnt[k] < 0 || s.shVOff[k] < 0 ||
            (size_t)s.shVOff[k] + (size_t)s.shVCnt[k] > s.verts.size())
        {
            err = "a shape's body or vertex range is out of bounds";
            return false;
        }
    for (size_t j = 0; j < J; j++)
        if (s.jointA[j] >= (int)N || s.jointB[j] < 0 || s.jointB[j] >= (int)N)
        {
            err = "a joint refers to a missing body";
            return false;
        }
    for (const uint64_t key : s.ignore)
        if ((size_t)(key >> 32) >= N || (size_t)(key & 0xffffffffu) >= N)
        {
            err = "a no-collide pair refers to a missing body";
            return false;
        }
    return true;
}

} // namespace scene_io_detail

inline bool saveScene2D(const Scene2D &scene, const char *path, const CaptureMeta2D &meta = CaptureMeta2D(),
                        std::string *err = nullptr)
{
    scene_io_detail::File file(fopen(path, "wb"));
    if (!file)
    {
        if (err)
            *err = std::string("cannot open ") + path + " for writing";
        return false;
    }
    scene_io_detail::Writer w{file.get()};
    w.raw(scene_io_detail::kMagic, sizeof(scene_io_detail::kMagic));
    w.str(meta.source);
    w.pod(meta.step);
    w.pod<uint32_t>(sizeof(SolverParams2D));
    w.pod(scene.params);
    w.pod(scene.cameraCenter);
    w.pod(scene.cameraHeight);
    visitScene2D(scene, [&](const auto &v) { w.array(v); });
    if (!w.ok || fflush(file.get()) != 0)
    {
        if (err)
            *err = std::string("write to ") + path + " failed";
        return false;
    }
    return true;
}

// Loads a capture into `scene` (replacing it). On failure returns false with the reason in *err
// and leaves `scene` untouched. A file whose solver-parameter block has another size loads with
// the default parameters, and *warning says so.
inline bool loadScene2D(Scene2D &scene, const char *path, CaptureMeta2D *meta = nullptr, std::string *err = nullptr,
                        std::string *warning = nullptr)
{
    auto fail = [&](const std::string &why) {
        if (err)
            *err = std::string(path) + ": " + why;
        return false;
    };
    scene_io_detail::File file(fopen(path, "rb"));
    if (!file)
        return fail("cannot open");
    fseek(file.get(), 0, SEEK_END);
    const long size = ftell(file.get());
    fseek(file.get(), 0, SEEK_SET);
    if (size < (long)sizeof(scene_io_detail::kMagic))
        return fail("not a scene capture");
    scene_io_detail::Reader r{file.get(), (uint64_t)size};
    char magic[8];
    r.raw(magic, sizeof(magic));
    if (memcmp(magic, scene_io_detail::kMagic, 6) != 0)
        return fail("not a scene capture");
    if (memcmp(magic, scene_io_detail::kMagic, 8) != 0)
        return fail("unsupported capture version");

    Scene2D s;
    CaptureMeta2D m;
    uint32_t paramBytes = 0;
    r.str(m.source);
    r.pod(m.step);
    r.pod(paramBytes);
    if (r.err.empty())
    {
        if (paramBytes == sizeof(SolverParams2D))
            r.pod(s.params);
        else
        {
            std::vector<char> skip(paramBytes);
            r.raw(skip.data(), skip.size());
            if (warning)
                *warning = "the solver parameters were saved by a different build; defaults are used";
        }
    }
    r.pod(s.cameraCenter);
    r.pod(s.cameraHeight);
    visitScene2D(s, [&](auto &v) { r.array(v); });
    if (!r.err.empty())
        return fail(r.err);
    std::string why;
    if (!scene_io_detail::validate(s, why))
        return fail(why);
    s.finalize();
    scene = std::move(s);
    if (meta)
        *meta = std::move(m);
    return true;
}

} // namespace avbd2d
