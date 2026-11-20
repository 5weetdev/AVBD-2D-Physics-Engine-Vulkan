// The C ABI of the 2D solver (avbd2d.h, ABI 3) on VkWorld2D + Scene2D.
//
// Shape of the layer: the host keeps slot tables (with free lists and generations) of bodies,
// shapes and joints. A created object is a record in its table and is "pending" until the next
// step_begin, which replays every pending body (with its shapes) and joint into the live world's
// authored Scene2D in one go and commits them (VkWorld2D::commitAuthored). Setters on a live
// object go to VkWorld2D's mutation queue; on a pending object they edit its record. Getters read
// VkWorld2D's host mirrors (the last completed step plus queued pose/velocity sets).
//
// Every entry point validates its handle or id first, so a null, never-created or destroyed world
// and a stale id are error codes and not crashes.


// Events, transforms, explosions, ray/shape casts and overlaps of the 2D C ABI.

#include "avbd2d_world.h"

extern "C"
{

    AVBD2D_API Avbd2dResult avbd2d_world_get_contact_events(Avbd2dWorld *world, Avbd2dContactEvents *out)
    {
        AVBD2D_GET_WORLD(world);
        NEED(out);
        out->beginEvents = w->beginEvents.empty() ? nullptr : w->beginEvents.data();
        out->endEvents = w->endEvents.empty() ? nullptr : w->endEvents.data();
        out->hitEvents = w->hitEvents.empty() ? nullptr : w->hitEvents.data();
        out->beginCount = (int32_t)w->beginEvents.size();
        out->endCount = (int32_t)w->endEvents.size();
        out->hitCount = (int32_t)w->hitEvents.size();
        return AVBD2D_OK;
    }

    AVBD2D_API Avbd2dResult avbd2d_world_get_sensor_events(Avbd2dWorld *world, Avbd2dSensorEvents *out)
    {
        AVBD2D_GET_WORLD(world);
        NEED(out);
        out->beginEvents = w->sensorBegin.empty() ? nullptr : w->sensorBegin.data();
        out->endEvents = w->sensorEnd.empty() ? nullptr : w->sensorEnd.data();
        out->beginCount = (int32_t)w->sensorBegin.size();
        out->endCount = (int32_t)w->sensorEnd.size();
        return AVBD2D_OK;
    }

    AVBD2D_API Avbd2dResult avbd2d_world_get_joint_events(Avbd2dWorld *world, Avbd2dJointEvents *out)
    {
        AVBD2D_GET_WORLD(world);
        NEED(out);
        out->jointEvents = w->jointEvents.empty() ? nullptr : w->jointEvents.data();
        out->count = (int32_t)w->jointEvents.size();
        return AVBD2D_OK;
    }

    AVBD2D_API Avbd2dResult avbd2d_world_get_transforms(Avbd2dWorld *world, Avbd2dBodyTransform *out, int32_t capacity,
                                                        int32_t *out_count)
    {
        AVBD2D_GET_WORLD(world);
        if (!out && capacity > 0)
            return AVBD2D_ERR_INVALID_ARG;
        int32_t n = 0;
        for (size_t i = 0; i < w->bodies.items.size(); i++)
        {
            BodyRec &b = w->bodies.items[i];
            if (!b.alive)
                continue;
            if (n < capacity)
            {
                const Pose p = poseOf(w, b);
                out[n].bodyId = bodyIdOf(w, (int)i);
                out[n].transform.p = A(p.origin());
                out[n].transform.q = live(b) ? rotOf(p.angle) : b.def.rotation;
            }
            n++;
        }
        if (out_count)
            *out_count = n;
        return AVBD2D_OK;
    }

    // ----- Explosions and queries ------------------------------------------------------------------
    AVBD2D_API Avbd2dResult avbd2d_world_explode(Avbd2dWorld *world, const Avbd2dExplosionDef *def)
    {
        AVBD2D_GET_WORLD(world);
        NEED(def);
        if (!finite2(def->position) || !(def->radius >= 0.0f) || !(def->falloff >= 0.0f) ||
            !finite1(def->impulsePerLength))
            return AVBD2D_ERR_INVALID_ARG;
        settle(w);
        return guarded(w, false,
                       [&]() -> Avbd2dResult
                       {
                           w->world->explode(V(def->position), def->radius, def->falloff, def->impulsePerLength);
                           return AVBD2D_OK;
                       });
    }

    AVBD2D_API Avbd2dResult avbd2d_world_cast_ray_closest(Avbd2dWorld *world, Avbd2dVec2 origin, Avbd2dVec2 translation,
                                                          Avbd2dQueryFilter filter, Avbd2dRayResult *out)
    {
        AVBD2D_GET_WORLD(world);
        NEED(out);
        if (!finite2(origin) || !finite2(translation))
            return AVBD2D_ERR_INVALID_ARG;
        *out = Avbd2dRayResult{};
        return guarded(w, false,
                       [&]() -> Avbd2dResult
                       {
                           w->world->castRay(V(origin), V(translation),
                                             [&](const RayHit2D &h) -> float
                                             {
                                                 int slot;
                                                 const ShapeRec *s = queryShape(w, h.shape, &slot);
                                                 if (!s || !passes(*s, filter))
                                                     return -1.0f;
                                                 out->hit = 1;
                                                 out->shapeId = shapeIdOf(w, slot);
                                                 out->point = A(h.point);
                                                 out->normal = A(h.normal);
                                                 out->fraction = h.fraction;
                                                 return h.fraction;
                                             });
                           return AVBD2D_OK;
                       });
    }

    AVBD2D_API Avbd2dResult avbd2d_world_cast_ray(Avbd2dWorld *world, Avbd2dVec2 origin, Avbd2dVec2 translation,
                                                  Avbd2dQueryFilter filter, Avbd2dCastResultFcn fcn, void *context)
    {
        AVBD2D_GET_WORLD(world);
        NEED(fcn);
        if (!finite2(origin) || !finite2(translation))
            return AVBD2D_ERR_INVALID_ARG;
        return guarded(w, false,
                       [&]() -> Avbd2dResult
                       {
                           w->world->castRay(V(origin), V(translation),
                                             [&](const RayHit2D &h) -> float
                                             {
                                                 int slot;
                                                 const ShapeRec *s = queryShape(w, h.shape, &slot);
                                                 if (!s || !passes(*s, filter))
                                                     return -1.0f;
                                                 return fcn(shapeIdOf(w, slot), A(h.point), A(h.normal), h.fraction,
                                                            context);
                                             });
                           return AVBD2D_OK;
                       });
    }

    AVBD2D_API Avbd2dResult avbd2d_world_cast_rays_batch(Avbd2dWorld *world, const Avbd2dRayQuery *rays, int32_t count,
                                                         Avbd2dQueryFilter filter, Avbd2dRayResult *out)
    {
        AVBD2D_GET_WORLD(world);
        if (count < 0)
            return AVBD2D_ERR_INVALID_ARG;
        if (count == 0)
            return AVBD2D_OK;
        NEED(rays);
        NEED(out);
        for (int32_t i = 0; i < count; i++)
            if (!finite2(rays[i].origin) || !finite2(rays[i].translation))
                return AVBD2D_ERR_INVALID_ARG;
        return guarded(w, false,
                       [&]() -> Avbd2dResult
                       {
                           std::vector<VkWorld2D::RayQuery2D> q((size_t)count);
                           std::vector<VkWorld2D::RayBatchHit2D> hits((size_t)count);
                           for (int32_t i = 0; i < count; i++)
                           {
                               q[(size_t)i].origin = V(rays[i].origin);
                               q[(size_t)i].translation = V(rays[i].translation);
                           }
                           w->world->castRaysBatch(q.data(), count, filter.categoryBits, filter.maskBits, hits.data());
                           for (int32_t i = 0; i < count; i++)
                           {
                               out[i] = Avbd2dRayResult{};
                               const VkWorld2D::RayBatchHit2D &h = hits[(size_t)i];
                               int slot;
                               const ShapeRec *s = h.shape >= 0 ? queryShape(w, h.shape, &slot) : nullptr;
                               if (h.shape >= 0 && (!s || !passes(*s, filter)))
                               {
                                   // A shape the ABI hides (destroyed before its body flushed): the answer for this
                                   // ray is the closest visible one, which only the host query can walk past.
                                   w->world->castRay(q[(size_t)i].origin, q[(size_t)i].translation,
                                                     [&](const RayHit2D &r) -> float
                                                     {
                                                         int sl;
                                                         const ShapeRec *sr = queryShape(w, r.shape, &sl);
                                                         if (!sr || !passes(*sr, filter))
                                                             return -1.0f;
                                                         out[i].hit = 1;
                                                         out[i].shapeId = shapeIdOf(w, sl);
                                                         out[i].point = A(r.point);
                                                         out[i].normal = A(r.normal);
                                                         out[i].fraction = r.fraction;
                                                         return r.fraction;
                                                     });
                                   continue;
                               }
                               if (h.shape < 0)
                                   continue;
                               out[i].hit = 1;
                               out[i].shapeId = shapeIdOf(w, slot);
                               out[i].point = A(h.point);
                               out[i].normal = A(h.normal);
                               out[i].fraction = h.fraction;
                           }
                           return AVBD2D_OK;
                       });
    }

    AVBD2D_API Avbd2dResult avbd2d_world_overlap_aabbs_batch(Avbd2dWorld *world, const Avbd2dAABB *aabbs, int32_t count,
                                                             Avbd2dQueryFilter filter, int32_t maxPerQuery,
                                                             Avbd2dShapeId *outShapeIds, int32_t *outCounts)
    {
        AVBD2D_GET_WORLD(world);
        if (count < 0 || maxPerQuery < 0)
            return AVBD2D_ERR_INVALID_ARG;
        if (count == 0)
            return AVBD2D_OK;
        NEED(aabbs);
        NEED(outCounts);
        if (maxPerQuery > 0)
            NEED(outShapeIds);
        for (int32_t i = 0; i < count; i++)
            if (!finite2(aabbs[i].lowerBound) || !finite2(aabbs[i].upperBound))
                return AVBD2D_ERR_INVALID_ARG;
        return guarded(w, false,
                       [&]() -> Avbd2dResult
                       {
                           const int per = std::max(maxPerQuery, 1);
                           std::vector<Vec2> lo((size_t)count), hi((size_t)count);
                           for (int32_t i = 0; i < count; i++)
                           {
                               lo[(size_t)i] = V(aabbs[i].lowerBound);
                               hi[(size_t)i] = V(aabbs[i].upperBound);
                           }
                           std::vector<int> ids((size_t)count * (size_t)per), counts((size_t)count);
                           w->world->overlapAabbsBatch(lo.data(), hi.data(), count, filter.categoryBits,
                                                       filter.maskBits, per, ids.data(), counts.data());
                           for (int32_t i = 0; i < count; i++)
                           {
                               int n = 0;
                               bool dropped = false;
                               const int got = std::min(counts[(size_t)i], per);
                               for (int k = 0; k < got; k++)
                               {
                                   int slot;
                                   const ShapeRec *s = queryShape(w, ids[(size_t)i * (size_t)per + (size_t)k], &slot);
                                   if (!s || !passes(*s, filter))
                                   {
                                       dropped = true;
                                       continue;
                                   }
                                   if (n < maxPerQuery)
                                       outShapeIds[(size_t)i * (size_t)maxPerQuery + (size_t)n] = shapeIdOf(w, slot);
                                   n++;
                               }
                               if (dropped && counts[(size_t)i] > per)
                               {
                                   // The device list was cut and some of it is hidden by the ABI: count this box on the host.
                                   n = 0;
                                   w->world->overlapAABB(
                                       lo[(size_t)i], hi[(size_t)i],
                                       [&](int shape) -> bool
                                       {
                                           int slot;
                                           const ShapeRec *s = queryShape(w, shape, &slot);
                                           if (!s || !passes(*s, filter))
                                               return true;
                                           if (n < maxPerQuery)
                                               outShapeIds[(size_t)i * (size_t)maxPerQuery + (size_t)n] =
                                                   shapeIdOf(w, slot);
                                           n++;
                                           return true;
                                       });
                               }
                               else if (counts[(size_t)i] > per)
                                   n += counts[(size_t)i] - per;
                               outCounts[i] = n;
                           }
                           return AVBD2D_OK;
                       });
    }

    AVBD2D_API Avbd2dResult avbd2d_world_overlap_aabb(Avbd2dWorld *world, Avbd2dAABB aabb, Avbd2dQueryFilter filter,
                                                      Avbd2dOverlapResultFcn fcn, void *context)
    {
        AVBD2D_GET_WORLD(world);
        NEED(fcn);
        if (!finite2(aabb.lowerBound) || !finite2(aabb.upperBound))
            return AVBD2D_ERR_INVALID_ARG;
        return guarded(w, false,
                       [&]() -> Avbd2dResult
                       {
                           w->world->overlapAABB(V(aabb.lowerBound), V(aabb.upperBound),
                                                 [&](int shape) -> bool
                                                 {
                                                     int slot;
                                                     const ShapeRec *s = queryShape(w, shape, &slot);
                                                     if (!s || !passes(*s, filter))
                                                         return true;
                                                     return fcn(shapeIdOf(w, slot), context) != 0;
                                                 });
                           return AVBD2D_OK;
                       });
    }

    AVBD2D_API Avbd2dResult avbd2d_world_overlap_shape(Avbd2dWorld *world, const Avbd2dShapeProxy *proxy,
                                                       Avbd2dQueryFilter filter, Avbd2dOverlapResultFcn fcn,
                                                       void *context)
    {
        AVBD2D_GET_WORLD(world);
        NEED(fcn);
        if (!validProxy(proxy))
            return AVBD2D_ERR_INVALID_ARG;
        const Proxy2D p = toProxy(*proxy);
        return guarded(w, false,
                       [&]() -> Avbd2dResult
                       {
                           w->world->overlapShape(p,
                                                  [&](int shape) -> bool
                                                  {
                                                      int slot;
                                                      const ShapeRec *s = queryShape(w, shape, &slot);
                                                      if (!s || !passes(*s, filter))
                                                          return true;
                                                      return fcn(shapeIdOf(w, slot), context) != 0;
                                                  });
                           return AVBD2D_OK;
                       });
    }

    AVBD2D_API Avbd2dResult avbd2d_world_cast_shape(Avbd2dWorld *world, const Avbd2dShapeProxy *proxy,
                                                    Avbd2dVec2 translation, Avbd2dQueryFilter filter,
                                                    Avbd2dCastResultFcn fcn, void *context)
    {
        AVBD2D_GET_WORLD(world);
        NEED(fcn);
        if (!validProxy(proxy) || !finite2(translation))
            return AVBD2D_ERR_INVALID_ARG;
        const Proxy2D p = toProxy(*proxy);
        return guarded(w, false,
                       [&]() -> Avbd2dResult
                       {
                           w->world->castShape(p, V(translation),
                                               [&](const RayHit2D &h) -> float
                                               {
                                                   int slot;
                                                   const ShapeRec *s = queryShape(w, h.shape, &slot);
                                                   if (!s || !passes(*s, filter))
                                                       return -1.0f;
                                                   return fcn(shapeIdOf(w, slot), A(h.point), A(h.normal), h.fraction,
                                                              context);
                                               });
                           return AVBD2D_OK;
                       });
    }
} // extern "C"
